#include <vector>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <tuple>
#include <random>
#include <map>
#include <set>
#include <dirent.h>
#include <sys/time.h>
#include <thread>
#include <stdio.h>
#include <malloc.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <Python.h>
#include <pybind11/pybind11.h>

#include "timer.h"
#include "log/log.h"
#include "queue.h"

using namespace std;
using intT = uint64_t;

#define PGSIZE 4096
#define IOSIZE 102458 * 24


std::mutex globalmutex;

struct Edge{
    uint64_t src;
    uint64_t dst;
};

std::tuple<std::vector<std::pair<intT, intT>>, intT>
load_edge_list(const std::string &file_path, char skip) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<intT, intT>> lines;
    lines.clear();
    std::ifstream ifs(file_path);
    intT max_vertex_id = 0;
    std::string tmp_str;
    intT src, dst;
    uint64_t line_count = 0;
    while (std::getline(ifs, tmp_str)) {
        line_count += 1;

        if (tmp_str[0] != skip) {
            std::stringstream ss(tmp_str);
            if (!(ss >> src >> dst)) {
                log_error("Cannot convert line %lu to edge.", line_count);
                exit(-1);
            }
            if (src > dst)
                std::swap(src, dst);
            lines.emplace_back(src, dst);

            if (src > max_vertex_id) {
                max_vertex_id = src;
            }
            if (dst > max_vertex_id) {
                max_vertex_id = dst;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    log_info("Load edge list file time: %.4f s", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);
}

// read edgelist
void Extractmap(string &input_folder_path, map<string, uint64_t> *&nodeRemap, uint64_t *&offset, uint64_t *&total, uint64_t &num_edges, int hashbucket)
{
    for (int i=0; i < hashbucket; ++ i) {
        offset[i] = 0;
        total[i] = 0;
    }
    DIR *dir;
    struct dirent *ent;
    int parallelism = thread::hardware_concurrency() - hashbucket;
    
    dir = opendir(input_folder_path.c_str());
	uint readFile = 0;
	string type_name("part");
    uint64_t totalFile = 1200;
    Queue<string> taskQueue;
    Queue<string> *indexMap = new Queue<string>[hashbucket]; 
    vector<thread> threads;
    threads.clear();
    vector<thread> mapthreads(hashbucket);
    mapthreads.clear();
    
    while ((ent = readdir(dir)) != NULL) {
        string input_file_path = input_folder_path + "/" + ent->d_name;
		string suffixStr = input_file_path.substr(input_file_path.find_last_of('/') + 1, min(4, (int)(input_file_path.length() - input_file_path.find_last_of('/'))));
        // cout << suffixStr << " " << suffixStr.size() << " " << endl;

		if (suffixStr.compare(type_name)) continue;
        readFile ++;
        printf("progress: %.2f%%\r", (double)readFile * 100 / totalFile);
        taskQueue.push(input_file_path);
    }
    for (int i = 0 ; i < parallelism; ++ i) {
        taskQueue.push("");
    }
    log_info("Task extraction finished! Total: %d", readFile);
    for (int i = 0; i < parallelism; ++ i) {
        threads.emplace_back([&](){
            while (true) {
                string input_file_path = taskQueue.pop();
                if (input_file_path == "") break;
                cout << input_file_path << endl;
                ifstream ifs(input_file_path);
                bool firstLine = true; // discard first line
                while (ifs.good())
                {
                    string tmp_str;
                    stringstream ss;
                    std::getline(ifs, tmp_str);
                    if (!ifs.good())
                        break;
                    if (firstLine) {
                        firstLine = false;
                        continue;
                    }
                    if (tmp_str[0] != '#')
                    {
                        ss.clear();
                        ss << tmp_str;
                        int64_t first, second;
                        double weight;
                        int8_t label;

                        ss >> first >> second >>  weight >> label;
                        // if (first == second)
                        //     continue;
                        indexMap[first % hashbucket].push(std::to_string(first));
                        indexMap[second % hashbucket].push(std::to_string(second));
                    }
                }
                ifs.close();
            }
        });
    }
    for (int i = 0; i < hashbucket; ++ i){
        mapthreads.emplace_back([&](int i) {
            while (true) {
                string node = indexMap[i].pop();
                if (node == "") break;
                if (nodeRemap[i].find(node) == nodeRemap[i].end()) {
                    nodeRemap[i][node] = offset[i];
                    offset[i] ++;
                }
                total[i] ++;    // the vertex is included in an edge 
            }
        }, i);
    }

    log_info("thread launched!");
    for (int i = 0; i < parallelism; ++ i) {
        threads[i].join();
    }
    log_info("Hash dispatch finish!");
    for (int j = 0; j < hashbucket; ++ j) {
        indexMap[j].push("");
    }
    for (int i = 0; i < hashbucket; ++ i) {
        mapthreads[i].join();
    }
    log_info("Finish!");
    for (int i = 0 ; i < hashbucket; ++ i) {
        cout << offset[i] << " ";
    }
    uint64_t dEdges = 0;
    for (int i = 0 ; i < hashbucket; ++ i) {
        dEdges += total[i];
    }
    log_info("double edges(total node) in uint64: %lu", dEdges);
    num_edges = dEdges / 2;
    assert(num_edges <= UINT64_MAX);
    log_info("total edge: %lu", num_edges);
}

void extractEdgelist(string input_folder_path, vector<std::pair<uint64_t, uint64_t>> &lines, uint64_t &max_ele, map<string, uint64_t> *&nodeRemap, uint64_t *&offset, int hashbucket) {

    DIR *dir;
    struct dirent *ent;
    uint readFile = 0;

    Queue<string> taskQueue;
    string type_name = "part";
    int parallelism = thread::hardware_concurrency();
    vector<thread> threads(parallelism);
    threads.clear();
    dir = opendir(input_folder_path.c_str());

    while ((ent = readdir(dir)) != NULL) {
        string input_file_path = input_folder_path + "/" + ent->d_name;
		string suffixStr = input_file_path.substr(input_file_path.find_last_of('/') + 1, min(4, (int)(input_file_path.length() - input_file_path.find_last_of('/'))));
        if (suffixStr.compare(type_name)) continue;
        readFile ++;
        taskQueue.push(input_file_path);
    }


    for (int i = 0 ; i < parallelism; ++ i) {
        taskQueue.push("finish");
    }
    log_info("finish reindexing... start extracting edge list");

    for (int ti = 0; ti < parallelism; ++ ti) {
        threads.emplace_back([&](int index){
            while (true) {
                string input_file_path = taskQueue.pop();
                if (input_file_path == "finish") {
                    break;
                }

                std::ifstream ifs(input_file_path);
                bool firstLine = true; // discard first line
                while (ifs.good())
                {
                    string tmp_str;
                    stringstream ss;
                    std::getline(ifs, tmp_str);
                    if (!ifs.good())
                        break;
                    if (firstLine) {
                        firstLine = false;
                        continue;
                    }
                    if (tmp_str[0] != '#')
                    {
                        ss.clear();
                        ss << tmp_str;
                        int64_t first, second;
                        double weight;
                        int8_t label;
                        
                        ss >> first >> second >> weight >> label;
                        // if (first == second) continue;
                        uint64_t src = nodeRemap[first%hashbucket][std::to_string(first)] + offset[first%hashbucket];
                        uint64_t dst = nodeRemap[second%hashbucket][std::to_string(second)] + offset[second%hashbucket];
                        // if (src > dst)
                        //     swap(src, dst);
                        if (src < 10)
                            log_info("%lu, %lu", src, dst);
                        if (max_ele < src)
                            max_ele = src;
                        if (max_ele < dst)
                            max_ele = dst;
                        
                        std::unique_lock<std::mutex> lock(globalmutex);
                        lines.emplace_back(src, dst);
                        lock.unlock();
                    }
                }
                ifs.close();
            }
        }, ti);
    }
    
    for (int i = 0; i < parallelism; ++ i) {
        threads[i].join();
    }

    log_info("begin writing edgelist ... ");
    ofstream edgelist_ofs("b.edgelist");
    for (uint64_t i = 0; i < lines.size(); i ++)
    {
        edgelist_ofs << lines[i].first << " " << lines[i].second << endl;
    }
    log_info("finish. ");

}

void WriteCSR(string &edgelistF, string &deg_output_file, string &adj_output_file, string &label_output_file, string &edgelist_output_file, string &weight_output_file, int num_labels, uint64_t &edge_num)
{

    uint64_t vertex_num = 0;
    struct stat st;
	assert(stat(edgelistF.c_str(), &st)==0);
	uint64_t fSize = st.st_size;

    int edgein = open(edgelistF.c_str(), O_RDONLY);
    uint64_t offset = 0;
    while(true)
    {  
        char * buffer = (char *)memalign(PGSIZE, IOSIZE);
        assert (buffer != NULL);
        memset (buffer, 0, IOSIZE);
        uint64_t bytes = pread(edgein, buffer, IOSIZE, offset);
        assert (bytes > 0);

        for (uint64_t pos = 0; pos + 2 * sizeof(uint64_t) <= bytes; pos += 2 * sizeof(uint64_t)) {
            uint64_t src = *(uint64_t *)(buffer + pos);
            uint64_t dst = *(uint64_t *)(buffer + pos + sizeof(uint64_t));
            if (src > vertex_num)
                vertex_num = src;
            if (dst > vertex_num)
                vertex_num = dst;
        }
        free(buffer);
        offset += bytes;
        if (offset >= fSize) break;
        fflush(stdout);
    }
    // edge_num = lines.size();
    uint64_t *degree_arr = new uint64_t [vertex_num + 1]{0};
    // vector<uint64_t> degree_arr(vertex_num, 0);
    uint64_t *offset_arr = new uint64_t [vertex_num]{0};
    // vector<vector<uint64_t>> matrix(vertex_num);
   
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_labels - 1);
    std::uniform_real_distribution<> dis2(0, 5.0);

    for (int i=0; i < 30; ++ i) 
        cout << degree_arr[i] << " " <<offset_arr[i] << " ";
    cout << endl;
    set<pair<uint64_t, uint64_t>> edgeSet;
    // FILE *edgein = fopen(edgelistF.c_str(), "r");
    ofstream deg_ofs(deg_output_file, ios::binary);
    // uint64_t line = 0;
    offset = 0;
    while(true)
    {  
        
        char * buffer = (char *)memalign(PGSIZE, IOSIZE);
        assert (buffer != NULL);
        memset (buffer, 0, IOSIZE);
        uint64_t bytes = pread(edgein, buffer, IOSIZE, offset);
        assert (bytes > 0);

        for (uint64_t pos = 0; pos + 2 * sizeof(uint64_t) <= bytes; pos += 2 * sizeof(uint64_t)) {
            uint64_t src = *(uint64_t *)(buffer + pos);
            uint64_t dst = *(uint64_t *)(buffer + pos + sizeof(uint64_t));
            if (edgeSet.find(make_pair(src, dst)) == edgeSet.end()) // avoid duplicate
                edgeSet.insert(make_pair(src, dst));
            else 
                continue;
            degree_arr[src]++;
            degree_arr[dst]++;
            
            offset_arr[src] ++;
            offset_arr[dst] ++;
        }
        free(buffer);
        offset += bytes;
        if (offset >= fSize) break;

       
    }
    
    close(edgein);

    cout << "Write phase 1..." << endl;
    edge_num = 0;   
    degree_arr[0] = 0;
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        edge_num += offset_arr[i];
        degree_arr[i + 1] = offset_arr[i] + degree_arr[i];
    }


    cout << edge_num << endl;

    cout << "begin write" << endl;

    deg_ofs.write(reinterpret_cast<const char *>(&vertex_num), sizeof(uint64_t));
    deg_ofs.write(reinterpret_cast<const char *>(&edge_num), sizeof(uint64_t));
    deg_ofs.write(reinterpret_cast<const char *>(degree_arr), (vertex_num + 1) * sizeof(uint64_t));
    deg_ofs.close();

    cout << "finish xadj write..." << endl;


    delete degree_arr;

    uint64_t ** matrix = new uint64_t *[vertex_num];
    uint64_t *outNum = new uint64_t[vertex_num];
    for (int i = 0; i < vertex_num; ++ i) {
        matrix[i] = new uint64_t [offset_arr[i]];
        outNum[i] = 0;
    }

    cout << "Allocation finish..." << endl;
    offset = 0;
    while(true)
    {  
        // uint64_t src, dst;
        // if (fread(&src, sizeof(uint64_t), 1, edgein) == 0)
        //     break;

        // if (fread(&dst, sizeof(uint64_t), 1, edgein) == 0)
        //     break;
        char * buffer = (char *)memalign(PGSIZE, IOSIZE);
        assert (buffer != NULL);
        memset (buffer, 0, IOSIZE);
        uint64_t bytes = pread(edgein, buffer, IOSIZE, offset);
        assert (bytes > 0);

        for (uint64_t pos = 0; pos + 2 * sizeof(uint64_t) <= bytes; pos += 2 * sizeof(uint64_t)) {
            uint64_t src = *(uint64_t *)(buffer + pos);
            uint64_t dst = *(uint64_t *)(buffer + pos + sizeof(uint64_t));
            
            if (edgeSet.find(make_pair(src, dst)) == edgeSet.end()) // avoid duplicate
                edgeSet.insert(make_pair(src, dst));
            else 
                continue;
            matrix[src][outNum[src]] = dst;
            matrix[dst][outNum[dst]] = src;
            outNum[src] ++;
            outNum[dst] ++;
            assert (outNum[src] <= offset_arr[src]);
            assert (outNum[dst] <= offset_arr[dst]);
            // matrix[src].emplace_back(dst);
		    // matrix[dst].emplace_back(src);
        }
        free(buffer);
        offset += bytes;
        if (offset >= fSize) break;
		// matrix[src].emplace_back(dst);
		// matrix[dst].emplace_back(src);
    }
    for (uint64_t src = 0; src < vertex_num ; ++ src)
        assert (outNum[src] == offset_arr[src]);
    close(edgein);


    ofstream adj_ofs(adj_output_file, ios::binary);
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        adj_ofs.write(reinterpret_cast<const char *>(matrix[i]), (offset_arr[i]) * sizeof(uint64_t));
    }
    // adj_ofs.write(reinterpret_cast<const char *>(offset_arr), (vertex_num + 1) * sizeof(uint64_t));
    adj_ofs.close();

    cout << "finish edge write..." << endl;
    ofstream edgelist_ofs(edgelist_output_file);
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        for (uint64_t j = 0; j < offset_arr[i]; ++ j)
        {
            edgelist_ofs << i << " " << matrix[i][j]<< endl;
        }
    }
    cout << "finish edgelist write..." << std::endl;
    edgelist_ofs.close();


    int ** labels = new int *[vertex_num];
    double ** weights = new double *[vertex_num];

    for (uint64_t i = 0; i < vertex_num; ++ i) {
        labels[i] = new int[offset_arr[i]];
        weights[i] = new double[offset_arr[i]];
        outNum[i] = 0;
    }
    cout << "Allocation finish..." << endl;


    offset = 0;
    while(true)
    {  
        char * buffer = (char *)memalign(PGSIZE, IOSIZE);
        assert (buffer != NULL);
        memset (buffer, 0, IOSIZE);
        uint64_t bytes = pread(edgein, buffer, IOSIZE, offset);
        assert (bytes > 0);
        for (uint64_t pos = 0; pos + 2 * sizeof(uint64_t) <= bytes; pos += 2 * sizeof(uint64_t)) {
            uint64_t src = *(uint64_t *)(buffer + pos);
            uint64_t dst = *(uint64_t *)(buffer + pos + sizeof(uint64_t));
            if (edgeSet.find(make_pair(src, dst)) == edgeSet.end()) // avoid duplicate
                edgeSet.insert(make_pair(src, dst));
            else 
                continue;
            int gen_label = dis(gen);
            labels[src][outNum[src]] = gen_label;
            labels[dst][outNum[dst]] = gen_label;
            // labels[src].emplace_back(gen_label);
            // labels[dst].emplace_back(gen_label);

            float gen_weight = dis2(gen);
            // weights[src].emplace_back(gen_weight);
            // weights[dst].emplace_back(gen_weight);
            weights[src][outNum[src]] = gen_weight;
            weights[dst][outNum[dst]] = gen_weight;
            outNum[src] ++;
            outNum[dst] ++;
        }
        free(buffer);
        offset += bytes;
        if (offset >= fSize) break;
        
    }
    for (uint64_t src = 0; src < vertex_num ; ++ src)
        assert (outNum[src] == offset_arr[src]);
    close(edgein);

    ofstream label_ofs(label_output_file, ios::binary);
    // label_ofs.write(reinterpret_cast<const char *>(&labels.front()), labels.size() * 4);
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        label_ofs.write(reinterpret_cast<const char *>(labels[i]), offset_arr[i] * sizeof(int));
    }
    cout << "finish label write..." << std::endl;
    label_ofs.close();

    ofstream weight_ofs(weight_output_file, ios::binary);
    // weight_ofs.write(reinterpret_cast<const char *>(&weights.front()), weights.size() * sizeof(float));
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        weight_ofs.write(reinterpret_cast<const char *>(weights[i]), offset_arr[i] * sizeof(float));
    }
    cout << "finish weight write..." << std::endl;
    weight_ofs.close();

}

void prepareFeature() {   
    // according the mapping, extracting the corresponding feature into a tensor and save it

}

void randomFeature(uint64_t total_nodes, uint64_t feature_dimensions) {   
    // according to the total node number and feature dimensions, generate the random feature tensor
    auto features = torch::rand({total_nodes, feature_dimensions});
    auto bytes = torch::jit::pickle_save(features);
    std::ofstream fout("x.zip", std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
    log_info("random generating feature finish...");
}

torch::Tensor load_from_edge_list(string input_folder_path) {
    uint64_t max_ele = 0;
    auto io_start = std::chrono::high_resolution_clock::now();
    int hashbucket = 64;
    map<string, uint64_t> * nodeRemap = new map<string, uint64_t>[hashbucket];
    uint64_t *total = new uint64_t [hashbucket];
    uint64_t *offset = new uint64_t [hashbucket];
    uint64_t num_edges;


    Extractmap(input_folder_path, nodeRemap, offset, total, num_edges, hashbucket);

    log_info("num edges: %d", num_edges);
    vector<std::pair<uint64_t, uint64_t>> lines;
    lines.reserve(num_edges);
    lines.clear();

    uint64_t *prefix = new uint64_t [hashbucket];
    for (int i = 0; i < hashbucket; ++ i) {
        prefix[i] = 0;
    }

    for (int i = 1; i < hashbucket; ++ i) {
        prefix[i] = prefix[i - 1] + offset[i - 1];
    }
    log_info("num nodes: %lu", prefix[hashbucket - 1] + offset[hashbucket - 1]);
    extractEdgelist(input_folder_path, lines, max_ele, nodeRemap, prefix, hashbucket);
    log_info("finish.");

    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto edge_tensor = torch::from_blob(lines.data(), {int64_t(lines.size()), 2}, opts).clone();
    return edge_tensor;
}





PYBIND11_MODULE(prepare_data, m) { // 'prepare_data' need to be consistent with the name in 'setup'
    m.def("load_from_edge_list", &load_from_edge_list, "loading graph from hdfs partition file", py::call_guard<py::gil_scoped_release>());
    m.def("generating_feature_random", &randomFeature, "generating feature randomly for testing", py::call_guard<py::gil_scoped_release>());
}

