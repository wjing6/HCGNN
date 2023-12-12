#include <ATen/ATen.h>
#include <Python.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <iostream>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <sys/stat.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <unistd.h>

#include "log/log.h"
#define WARP_SIZE 32
#define ALIGNMENT 4096

#define gpuErrorcheck(ans)                                                     \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}

struct not_exist_ {
    __host__ __device__ bool operator()(const int x) { return (x == -1); }
};

enum GPU_Mode { UVA, ZERO_COPY, GPUDirect, CUDAMemcpy };

__global__ void gather_tensor_UVA(int64_t *node_idx, int64_t *cache_indices,
                                  int64_t *tmp_indices, char *feature_cache,
                                  char *feature_tmp, int feature_dim,
                                  int64_t num_idx, char *res)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int warp_step = step / WARP_SIZE;

    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % WARP_SIZE;

    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;

    int64_t feature_size = feature_dim * 4;  // default: float32 - 4B
    unsigned int local_start = thread_start;
    while (warp_start < num_idx) {
        local_start = thread_start;
        int64_t index = node_idx[warp_start];
        dst_copy_start = warp_start * feature_size;
        if (cache_indices[index] != -1) {
            src_copy_start = cache_indices[index] * feature_size;
            for (; local_start < feature_size; local_start += WARP_SIZE) {
                res[dst_copy_start + local_start] =
                    feature_cache[src_copy_start + local_start];
            }
        } else if (tmp_indices[index] != -1) {
            src_copy_start = tmp_indices[index] * feature_size;
            for (; local_start < feature_size; local_start += WARP_SIZE) {
                res[dst_copy_start + local_start] =
                    feature_tmp[src_copy_start + local_start];
            }
        } else {
            std::printf("something error...\n");
        }
        warp_start += warp_step;
    }
}

class CUDA_Gather
{
  private:
    int64_t *feature_cache_indice;  // now the cache is static
    char *feature_cache;
    std::string feature_file;
    std::string feature_indice_file;
    int64_t num_nodes;
    int feature_dim;
    GPU_Mode mode;

  public:
    CUDA_Gather(std::string f_feature, std::string f_feature_indice,
                int64_t num_nodes, int feature_dim, GPU_Mode mode = UVA)
        : feature_file(f_feature),
          feature_indice_file(f_feature_indice),
          num_nodes(num_nodes),
          feature_dim(feature_dim),
          mode(mode)
    {
    }
    torch::Tensor gather_from_gpu(torch::Tensor node_idx)
    {
        int feature_fd = open(this->feature_file.c_str(), O_RDONLY | O_DIRECT);
        int64_t feature_size = this->feature_dim * sizeof(float);

        int num_idx = node_idx.numel();
        auto idx_data = node_idx.data_ptr<int64_t>();

        char *feature_tmp = nullptr;
        int64_t *tmp_indices = nullptr;
        if (this->mode == UVA) {
            gpuErrorcheck(cudaMallocManaged(&tmp_indices,
                                            this->num_nodes * sizeof(int64_t)));
            gpuErrorcheck(
                cudaMemset(tmp_indices, -1, this->num_nodes * sizeof(int64_t)));
        } else {
            log_error("error, now only supporting UVA..");
        }

#pragma omp parallel for num_threads(atoi(getenv("GATHER_NUM_THREADS")))
        for (int i = 0; i < num_idx; ++i) {
            int idx = idx_data[i];
            if (this->feature_cache_indice[idx] == -1) {
                // #pragma omp critical {
                tmp_indices[idx] = idx;
                // }
            }
        }
        int64_t not_in_cache =
            num_nodes - thrust::count(tmp_indices, tmp_indices + num_nodes, -1);
        int64_t *tmp = (int64_t *)malloc(not_in_cache * sizeof(int64_t));
        thrust::remove_copy_if(thrust::host, tmp_indices, tmp_indices + num_nodes,
                               tmp, not_exist_());
        int64_t *H = (int64_t *)malloc(not_in_cache * sizeof(int64_t));
        thrust::sequence(H, H + not_in_cache);

        log_info("the feature not in cache: %d", not_in_cache);
#pragma omp parallel for num_threads(atoi(getenv("GATHER_NUM_THREADS")))
        for (int i = 0; i < not_in_cache; ++i) { tmp_indices[tmp[i]] = H[i]; }

        free(tmp);
        log_info("calculate finish..");

        if (not_in_cache > 0) {
            log_info("alloc memory for features not in cache..");
            if (mode == UVA) {
                gpuErrorcheck(cudaMallocManaged(&feature_tmp,
                                                not_in_cache * feature_size));
            } else {
                log_error("error, now only supporting UVA..");
            }
            log_info("alloc finish..");
            int start = 0;
            float *read_buffer = (float *)aligned_alloc(
                ALIGNMENT, ALIGNMENT * 2 * atoi(getenv("GATHER_NUM_THREADS")));
#pragma omp parallel for num_threads(atoi(getenv("GATHER_NUM_THREADS")))
            for (int i = 0; i < num_idx; ++i) {
                int idx = idx_data[i];
                if (this->feature_cache_indice[idx] == -1) {
                    int64_t offset = idx * feature_size;
                    int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
                    int64_t residual = offset - aligned_offset;
                    int64_t read_size;
                    if (residual + feature_size > ALIGNMENT) {
                        read_size = ALIGNMENT * 2;
                    } else {
                        read_size = ALIGNMENT;
                    }
                    if (pread(feature_fd,
                              read_buffer +
                                  (ALIGNMENT * 2 * omp_get_thread_num()) /
                                      sizeof(float),
                              read_size, aligned_offset) == -1) {
                        log_error("ERROR: %s\n", strerror(errno));
                    }
                    memcpy(feature_tmp + feature_size * tmp_indices[idx],
                           read_buffer + (ALIGNMENT * 2 * omp_get_thread_num() +
                                          residual) /
                                             sizeof(float),
                           feature_size);
                }
            }
            free(read_buffer);
        }
        log_info("prepare aux cache finish..");
        char *res = nullptr;
        gpuErrorcheck(cudaMalloc(&res, feature_size * num_idx));
        int64_t *dev_node_idx;
        gpuErrorcheck(cudaMalloc(&dev_node_idx, sizeof(int64_t) * num_idx));
        gpuErrorcheck(cudaMemcpy(dev_node_idx, idx_data,
                                 num_idx * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));

        log_info("GPU gather start..");
        int blockSize = 512;
        gather_tensor_UVA<<<(num_idx - 1) / blockSize + 1, blockSize>>>(
            dev_node_idx, this->feature_cache_indice, tmp_indices,
            this->feature_cache, feature_tmp, this->feature_dim, num_idx, res);

        gpuErrorcheck(cudaDeviceSynchronize());
        gpuErrorcheck(cudaPeekAtLastError());
        log_info("GPU gather finish..");
        auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .layout(torch::kStrided)
                           .device(torch::kCUDA)
                           .requires_grad(false);

        auto result = torch::from_blob(res, {num_idx, feature_dim}, options);

        // need delete something..
        if (not_in_cache > 0) {
            gpuErrorcheck(cudaFree(feature_tmp));
            gpuErrorcheck(cudaFree(tmp_indices));
        }
        // gpuErrorcheck(cudaFree(res));
        gpuErrorcheck(cudaFree(dev_node_idx));
        close(feature_fd);
        return result;
    }

    void load_static_feature_cache_from_scratch()
    {
        // struct stat statbuf;
        // stat(this->feature_indice_file.c_str(), &statbuf);
        // int64_t file_size = (int64_t)statbuf.st_size;

        int feature_indice_fd =
            open((this->feature_indice_file).c_str(), O_RDONLY);
        int feature_fd =
            open((this->feature_file).c_str(), O_RDONLY | O_DIRECT);
        log_info("%s, %s", this->feature_indice_file.c_str(),
                 this->feature_file.c_str());
        if (feature_indice_fd < 0 || feature_fd < 0) {
            log_error("open failed.");
            return;
        }
        int64_t feature_size = this->feature_dim * sizeof(float);
        // int64_t feature_num = int64_t(file_size / sizeof(int64_t));
        int64_t feature_num = 9765625;  // papers100M
        gpuErrorcheck(cudaMallocManaged(&(this->feature_cache_indice),
                                        this->num_nodes * sizeof(int64_t)));

        gpuErrorcheck(cudaMallocManaged(&(this->feature_cache),
                                        feature_num * feature_size));
        gpuErrorcheck(cudaMemset(this->feature_cache_indice, -1,
                                 this->num_nodes * sizeof(int64_t)));

        int64_t *indice_buffer =
            (int64_t *)malloc(feature_num * sizeof(int64_t));
        log_info("indice finish");
        float *feature_buffer = (float *)aligned_alloc(
            ALIGNMENT, ALIGNMENT * 2 * atoi(getenv("GATHER_NUM_THREADS")));

        log_info("mallocing finish, feature number: %d, feature dimension: %d",
                 feature_num, this->feature_dim);

        if (pread(feature_indice_fd, indice_buffer,
                  feature_num * sizeof(int64_t), 0) < 0) {
            log_error("There is an error when reading..");
            free(indice_buffer);
            free(feature_buffer);
            close(feature_indice_fd);
            close(feature_fd);
            return;
        }
        log_info("reading indice finish.. start gathering caching.. ");

#pragma omp parallel for num_threads(atoi(getenv("GATHER_NUM_THREADS")))
        for (int i = 0; i < feature_num; ++i) {
            this->feature_cache_indice[indice_buffer[i]] = i;

            int64_t offset = indice_buffer[i] * feature_size;
            int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
            int64_t residual = offset - aligned_offset;
            int64_t read_size;
            if (residual + feature_size > ALIGNMENT) {
                read_size = ALIGNMENT * 2;
            } else {
                read_size = ALIGNMENT;
            }
            if (pread(feature_fd,
                      feature_buffer + (ALIGNMENT * 2 * omp_get_thread_num()) /
                                           sizeof(float),
                      read_size, aligned_offset) == -1) {
                log_error("ERROR: %s\n", strerror(errno));
            }
            memcpy(this->feature_cache + feature_size * i,
                   feature_buffer +
                       (ALIGNMENT * 2 * omp_get_thread_num() + residual) /
                           sizeof(float),
                   feature_size);
        }

        free(indice_buffer);
        free(feature_buffer);
        close(feature_indice_fd);
        close(feature_fd);
        log_info("initialize cache finished..");
    }

    void load_static_feature_from_file(std::string feature_cache_file)
    {
        struct stat statbuf;
        stat(feature_cache_file.c_str(), &statbuf);
        int64_t file_size = (int64_t)statbuf.st_size;

        int feature_indice_fd =
            open((this->feature_indice_file).c_str(), O_RDONLY);
        int feature_cache_fd =
            open((feature_cache_file).c_str(), O_RDONLY | O_DIRECT);
        if (feature_indice_fd < 0 || feature_cache_fd < 0) {
            log_error("open failed.. Please make sure that the file exists");
            return;
        }

        int64_t feature_size = this->feature_dim * sizeof(float);
        int64_t feature_num = int64_t(file_size / feature_size);
        // int64_t feature_num = 9765625;
        // assert(feature_num == 9765625);
        log_info("cache memory alloc start..");
        gpuErrorcheck(cudaMallocManaged(&(this->feature_cache_indice),
                                        this->num_nodes * sizeof(int64_t)));

        gpuErrorcheck(cudaMallocManaged(&(this->feature_cache),
                                        feature_num * feature_size));
        gpuErrorcheck(cudaMemset(this->feature_cache_indice, -1,
                                 this->num_nodes * sizeof(int64_t)));

        log_info("cache memory alloc finish..");
        int64_t *indice_buffer =
            (int64_t *)malloc(feature_num * sizeof(int64_t));

        if (pread(feature_indice_fd, indice_buffer,
                  feature_num * sizeof(int64_t), 0) < 0) {
            log_error("There is an error when reading..");
            free(indice_buffer);
            close(feature_indice_fd);
            close(feature_cache_fd);
            return;
        }
        log_info("reading indice finish.. start gathering caching.. ");

#pragma omp parallel for num_threads(atoi(getenv("GATHER_NUM_THREADS")))
        for (int i = 0; i < feature_num; ++i) {
            this->feature_cache_indice[indice_buffer[i]] = i;
        }

        if (pread(feature_cache_fd, this->feature_cache,
                  feature_num * feature_size, 0) < 0) {
            log_error("There is an error when reading..");
            free(indice_buffer);
            close(feature_indice_fd);
            close(feature_cache_fd);
            return;
        }

        free(indice_buffer);
        close(feature_indice_fd);
        close(feature_cache_fd);
        log_info("reading cache from file finished..");
    }

    // free the memory and make load only once
    ~CUDA_Gather()
    {
        gpuErrorcheck(cudaFree(this->feature_cache));
        gpuErrorcheck(cudaFree(this->feature_cache_indice));
        log_info("free finish..");
    }
};

PYBIND11_MODULE(gather_gpu, m)
{
    py::class_<CUDA_Gather>(m, "CUDA_Gather")
        .def(py::init<std::string, std::string, int64_t, int>())
        .def("gather", &CUDA_Gather::gather_from_gpu,
             py::call_guard<py::gil_scoped_release>())
        .def("init_static_from_scratch",
             &CUDA_Gather::load_static_feature_cache_from_scratch,
             py::call_guard<py::gil_scoped_release>())
        .def("init_static_from_file",
             &CUDA_Gather::load_static_feature_from_file,
             py::call_guard<py::gil_scoped_release>());
}
