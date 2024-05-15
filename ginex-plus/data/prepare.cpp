#include <ATen/ATen.h>
#include <Python.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <fstream>
#include <sstream>
#include <inttypes.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <thread>
#include <map>
#include <set>
#include <tuple>
#include <random>
#include <mutex>
#include <dirent.h>

#include "../utils/log/log.h"
#include "../utils/queue.h"
// prepare from the edge-list
// the format is like:
// src_id   dst_id
//   0  ,  1
//   0  ,  2
// .................
#define IOSIZE (1048576 * 24 * 2)
#define PGSIZE 4096
using intT = int64_t;

std::mutex globalmutex;


void store_binary_from_edge_list(std::vector<std::pair<intT, intT>> &lines,
                                 std::string outfile)
{
    std::ofstream edgelist_ofs(outfile, std::ios::binary);
    log_info("begin converting .edgelist to .bin");
    int64_t size = lines.size() * sizeof(intT) * 2;
    edgelist_ofs.write(reinterpret_cast<char *>(lines.data()), size);
    edgelist_ofs.close();
    log_info("store .binary finish");
}

std::tuple<torch::Tensor, intT> load_edge_list(const std::string &file_path,
                                               char skip, bool store = false)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<intT, intT>> lines;
    lines.clear();
    std::ifstream ifs(file_path);
    intT max_vertex_id = 0;
    std::string tmp_str;
    intT src, dst;
    char delimiter;
    uint64_t line_count = 0;
    while (std::getline(ifs, tmp_str)) {
        line_count += 1;

        if (tmp_str[0] != skip) {
            std::stringstream ss(tmp_str);
            if (!(ss >> src >> delimiter >> dst)) {
                log_error("Cannot convert line %lu to edge.", line_count);
                exit(-1);
            }
            if (line_count == 1) {
                log_info("src: %lu, dst: %lu, delimiter: %c", src, dst,
                         delimiter);
            }
            lines.emplace_back(src, dst);

            if (src > max_vertex_id) { max_vertex_id = src; }
            if (dst > max_vertex_id) { max_vertex_id = dst; }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    log_info("Load edge list file time: %.4f s",
             std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                     .count() /
                 1000.0);
    log_info("dataset: %s, num_nodes: %lu",
             (file_path.substr(file_path.find_last_of('/') + 1)).c_str(),
             max_vertex_id + 1);
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto edge_tensor =
        torch::from_blob(lines.data(), {int64_t(lines.size()), 2}, opts)
            .clone();

    if (store) {
        log_info("store binary for next acceleration");
        std::string outfile_path =
            file_path.substr(0, file_path.find_last_of('.'));
        outfile_path = outfile_path + ".bin";
        log_info("save path: %s", outfile_path.c_str());
        store_binary_from_edge_list(lines, outfile_path);
    }

    return std::make_tuple(edge_tensor, max_vertex_id + 1);
}

std::tuple<torch::Tensor, intT>
load_edge_list_from_binary(const std::string file_path)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<intT, intT>> lines;
    lines.clear();

    int fd = open(file_path.c_str(), O_RDONLY); 
    // 如果添加O_DIRECT, 会出错; 因此开启O_DIRECT之后, 读取大小必须是4K的整数倍
    if (fd < 0) { log_error("file not exist"); }
    log_info("open %s complete", file_path.c_str());
    intT max_vertex_id = 0;
    intT src, dst;
    long read_bytes = 0;
    // log_info("file size: %lu", file_size);
    char *buffer = (char *)malloc(IOSIZE);
    log_info("malloc finish");
    while (1) {
        long bytes = pread(fd, buffer, IOSIZE, read_bytes);
        if (bytes < 0) {
            log_error("error.");
        }
        if (bytes == 0) break;
        for (uint64_t pos = 0; pos < bytes; pos += 2 * sizeof(int64_t)) {
            src = *(intT *)(buffer + pos);
            dst = *(intT *)(buffer + pos + sizeof(int64_t));
            if (read_bytes == 0 && pos == 0) {
                log_info("src: %lld, dst: %lld", src, dst);
            }
            if (src > max_vertex_id)
                max_vertex_id = src;
            if (dst > max_vertex_id)
                max_vertex_id = dst;
            lines.emplace_back(src, dst);
        }
        read_bytes += bytes;
    }
    auto end = std::chrono::high_resolution_clock::now();
    log_info("Load edge list file time: %.4f s",
             std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                     .count() /
                 1000.0);

    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto edge_tensor =
        torch::from_blob(lines.data(), {int64_t(lines.size()), 2}, opts)
            .clone();
    free(buffer);
    close(fd);
    return std::make_tuple(edge_tensor, max_vertex_id + 1);
}


// extract edgelist from hdfs-split-file, multi-thread to accelerate preprocessing
void Extractmap(std::string &input_folder_path, std::map<std::string, int64_t> *&nodeRemap, 
                uint64_t *&offset, uint64_t *&total, uint64_t &num_edges, 
                int hashbucket, bool delimiter)
{
    for (int i=0; i < hashbucket; ++ i) {
        offset[i] = 0;
        total[i] = 0;
    }
    DIR *dir;
    struct dirent *ent;
    int parallelism = std::thread::hardware_concurrency() - hashbucket;
    
    dir = opendir(input_folder_path.c_str());
	uint readFile = 0;
	std::string type_name("part");
    uint64_t totalFile = 100000; //
    // getFiles(input_folder_path, &totalFile);
    log_info("In folder %s, the file num is %lu", input_folder_path.c_str(), totalFile);
    Queue<std::string> taskQueue(totalFile);
    Queue<std::string> *indexMap = new Queue<std::string>[hashbucket]; 
    std::vector<std::thread> threads;
    threads.clear();
    std::vector<std::thread> mapthreads(hashbucket);
    mapthreads.clear();
    
    while ((ent = readdir(dir)) != NULL) {
        std::string input_file_path = input_folder_path + "/" + ent->d_name;
		std::string suffixStr = input_file_path.substr(input_file_path.find_last_of('/') + 1, std::min(4, (int)(input_file_path.length() - input_file_path.find_last_of('/'))));
        
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
                std::string input_file_path = taskQueue.pop();
                if (input_file_path == "") break;
                std::ifstream ifs(input_file_path);
                bool firstLine = true; // discard first line
                bool printDebug = true;
                while (ifs.good())
                {
                    std::string tmp_str;
                    std::stringstream ss;
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
                        if (!delimiter)
                            ss >> first >> second >>  weight >> label;
                        else {
                            char del;
                            ss >> first >> del >> second >> del >> weight >> del >> label;
                        }
                        if (first == second)
                            continue;
                        if (printDebug) {
                            log_info("%s, src: %lu, dst: %lu, weight: %f, label: 0x%02x", input_file_path.c_str(), first, second, weight, label);
                            printDebug = false;
                        }
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
                std::string node = indexMap[i].pop();
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
        log_info("%lu ", offset[i]);
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

void extractEdgelist(std::string input_folder_path, std::vector<std::pair<int64_t, int64_t>> &lines, 
                     int64_t &max_ele, std::map<std::string, int64_t> *&nodeRemap, uint64_t *&offset, 
                     int hashbucket, bool store, bool delimiter) {

    DIR *dir;
    struct dirent *ent;
    uint readFile = 0;

    Queue<std::string> taskQueue;
    std::string type_name = "part";
    int parallelism = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(parallelism);
    threads.clear();
    dir = opendir(input_folder_path.c_str());

    while ((ent = readdir(dir)) != NULL) {
        std::string input_file_path = input_folder_path + "/" + ent->d_name;
		std::string suffixStr = input_file_path.substr(input_file_path.find_last_of('/') + 1, std::min(4, (int)(input_file_path.length() - input_file_path.find_last_of('/'))));
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
                std::string input_file_path = taskQueue.pop();
                if (input_file_path == "finish") {
                    break;
                }

                std::ifstream ifs(input_file_path);
                bool firstLine = true; // discard first line
                while (ifs.good())
                {
                    std::string tmp_str;
                    std::stringstream ss;
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
                        
                        if (!delimiter)
                            ss >> first >> second >>  weight >> label;
                        else {
                            char del;
                            ss >> first >> del >> second >> del >> weight >> del >> label;
                        }
                        if (first == second) continue;
                        int64_t src = nodeRemap[first%hashbucket][std::to_string(first)] + offset[first%hashbucket];
                        int64_t dst = nodeRemap[second%hashbucket][std::to_string(second)] + offset[second%hashbucket];
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


    if (store) {
        log_info("store binary for next acceleration");
        std::string outfile_path = input_folder_path + "/../b.bin";
        log_info("save path: %s", outfile_path.c_str());
        store_binary_from_edge_list(lines, outfile_path);
    }
    // saving binary rather than edgelist
    
    // log_info("begin writing edgelist ... ");
    // std::ofstream edgelist_ofs("b.edgelist");
    // for (uint64_t i = 0; i < lines.size(); i ++)
    // {
    //     edgelist_ofs << lines[i].first << " " << lines[i].second << std::endl;
    // }
    // log_info("finish. ");

}

void WriteCSR(std::string &edgelistF, std::string &deg_output_file, std::string &adj_output_file, 
              std::string &label_output_file, std::string &edgelist_output_file, std::string &weight_output_file, 
              int num_labels, uint64_t &edge_num)
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
    
    uint64_t *degree_arr = new uint64_t [vertex_num + 1]{0};
    // vector<uint64_t> degree_arr(vertex_num, 0);
    uint64_t *offset_arr = new uint64_t [vertex_num]{0};
    // vector<vector<uint64_t>> matrix(vertex_num);
   
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_labels - 1);
    std::uniform_real_distribution<> dis2(0, 5.0);

    for (int i=0; i < 30; ++ i) 
        log_info("%lu, %lu ", degree_arr[i] , offset_arr[i]);
    std::set<std::pair<uint64_t, uint64_t>> edgeSet;
    // FILE *edgein = fopen(edgelistF.c_str(), "r");
    std::ofstream deg_ofs(deg_output_file, std::ios::binary);
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
            if (edgeSet.find(std::make_pair(src, dst)) == edgeSet.end()) // avoid duplicate
                edgeSet.insert(std::make_pair(src, dst));
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

    log_info("Write phase 1...");
    edge_num = 0;   
    degree_arr[0] = 0;
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        edge_num += offset_arr[i];
        degree_arr[i + 1] = offset_arr[i] + degree_arr[i];
    }

    deg_ofs.write(reinterpret_cast<const char *>(&vertex_num), sizeof(uint64_t));
    deg_ofs.write(reinterpret_cast<const char *>(&edge_num), sizeof(uint64_t));
    deg_ofs.write(reinterpret_cast<const char *>(degree_arr), (vertex_num + 1) * sizeof(uint64_t));
    deg_ofs.close();

    log_info("finish xadj write...");


    delete degree_arr;

    uint64_t ** matrix = new uint64_t *[vertex_num];
    uint64_t *outNum = new uint64_t[vertex_num];
    for (int i = 0; i < vertex_num; ++ i) {
        matrix[i] = new uint64_t [offset_arr[i]];
        outNum[i] = 0;
    }

    log_info( "Allocation finish..." );
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
            
            if (edgeSet.find(std::make_pair(src, dst)) == edgeSet.end()) // avoid duplicate
                edgeSet.insert(std::make_pair(src, dst));
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


    std::ofstream adj_ofs(adj_output_file, std::ios::binary);
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        adj_ofs.write(reinterpret_cast<const char *>(matrix[i]), (offset_arr[i]) * sizeof(uint64_t));
    }
    // adj_ofs.write(reinterpret_cast<const char *>(offset_arr), (vertex_num + 1) * sizeof(uint64_t));
    adj_ofs.close();

    log_info("finish edge write...");
    std::ofstream edgelist_ofs(edgelist_output_file);
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        for (uint64_t j = 0; j < offset_arr[i]; ++ j)
        {
            edgelist_ofs << i << " " << matrix[i][j]<< std::endl;
        }
    }
    log_info("finish edgelist write...");
    edgelist_ofs.close();


    int ** labels = new int *[vertex_num];
    double ** weights = new double *[vertex_num];

    for (uint64_t i = 0; i < vertex_num; ++ i) {
        labels[i] = new int[offset_arr[i]];
        weights[i] = new double[offset_arr[i]];
        outNum[i] = 0;
    }
    log_info("Allocation finish...");


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
            if (edgeSet.find(std::make_pair(src, dst)) == edgeSet.end()) // avoid duplicate
                edgeSet.insert(std::make_pair(src, dst));
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

    std::ofstream label_ofs(label_output_file, std::ios::binary);
    // label_ofs.write(reinterpret_cast<const char *>(&labels.front()), labels.size() * 4);
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        label_ofs.write(reinterpret_cast<const char *>(labels[i]), offset_arr[i] * sizeof(int));
    }
    log_info("finish label write...");
    label_ofs.close();

    std::ofstream weight_ofs(weight_output_file, std::ios::binary);
    // weight_ofs.write(reinterpret_cast<const char *>(&weights.front()), weights.size() * sizeof(float));
    for (uint64_t i = 0; i < vertex_num; i++)
    {
        weight_ofs.write(reinterpret_cast<const char *>(weights[i]), offset_arr[i] * sizeof(float));
    }
    log_info("finish weight write...");
    weight_ofs.close();

}

void prepareFeature() {   
    // according the mapping, extracting the corresponding feature into a tensor and save it

}

void randomFeature(std::string dataset, uint64_t total_nodes, uint64_t train_num, uint64_t feature_dim) {   
    // according to the total node number and feature dimensions, generate the random feature tensor
    auto features = torch::rand({total_nodes, feature_dim}, torch::dtype(torch::kFloat32));
    auto bytes_feature = torch::jit::pickle_save(features);
    std::ofstream fout_feature("features.dat", std::ios::out | std::ios::binary);
    fout_feature.write(bytes_feature.data(), bytes_feature.size());
    fout_feature.close();

    auto labels = torch::rand({train_num, 1}, torch::dtype(torch::kFloat32));
    auto bytes = torch::jit::pickle_save(labels);
    std::ofstream fout_label("labels.dat", std::ios::out | std::ios::binary);
    fout_label.write(bytes.data(), bytes.size());
    fout_label.close();
    log_info("random generating feature finish...");
}

std::tuple<torch::Tensor, int64_t>
load_from_part_file(std::string input_folder_path, bool store = false, bool delimiter = false) {
    int64_t max_ele = 0;
    int hashbucket = 64;
    std::map<std::string, int64_t> * nodeRemap = new std::map<std::string, int64_t>[hashbucket];
    uint64_t *total = new uint64_t [hashbucket];
    uint64_t *offset = new uint64_t [hashbucket];
    uint64_t num_edges;
    Extractmap(input_folder_path, nodeRemap, offset, total, num_edges, hashbucket, delimiter);

    log_info("num edges: %lu", num_edges);
    std::vector<std::pair<int64_t, int64_t>> lines;
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
    extractEdgelist(input_folder_path, lines, max_ele, nodeRemap, prefix, hashbucket, store, delimiter);
    log_info("finish.");

    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto edge_tensor = torch::from_blob(lines.data(), {int64_t(lines.size()), 2}, opts).clone();
    // return the number nodes
    return std::make_tuple(edge_tensor, max_ele + 1);
}

PYBIND11_MODULE(prepare_data_from_scratch, m)
{
    m.def("load_edge_list", &load_edge_list, "load edge list",
          py::call_guard<py::gil_scoped_release>());
    m.def("load_edge_list_from_binary", &load_edge_list_from_binary,
          "load edge list from binary",
          py::call_guard<py::gil_scoped_release>());
    m.def("load_from_part_file", &load_from_part_file, 
          "extract edge list from split files(hdfs)",
          py::call_guard<py::gil_scoped_release>());
    m.def("randomFeature", &randomFeature,
          "generate the features randomly",
          py::call_guard<py::gil_scoped_release>());
}
