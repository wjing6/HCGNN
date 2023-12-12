#include <ATen/ATen.h>
#include <Python.h>
#include <aio.h>
#include <chrono>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <pthread.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <unistd.h>
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
// TODO: using <aiocb.h> to make async read, to test the performance

class Timer
{
  public:
    Timer() : beg_(clock_::now()) {}

    void reset() { beg_ = clock_::now(); }

    double elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   clock_::now() - beg_)
            .count();
    }

    double elapsed_and_reset()
    {
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             clock_::now() - beg_)
                             .count();
        beg_ = clock_::now();
        return elapsed;
    }

  private:
    typedef std::chrono::high_resolution_clock clock_;
    std::chrono::time_point<clock_> beg_;
};

torch::Tensor gather_mmap(torch::Tensor features, torch::Tensor idx,
                          int64_t feature_dim)
{

    // open file
    int64_t feature_size = feature_dim * sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();
    float *result_buffer =
        (float *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

    auto features_data = features.data_ptr<float>();
    auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for num_threads(atoi(getenv("GINEX_NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;

        i = idx_data[n];
        memcpy(result_buffer + feature_dim * n, features_data + i * feature_dim,
               feature_size);
    }

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result =
        torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

    return result;
}

std::tuple<torch::Tensor, int64_t>
gather_ginex(std::string feature_file, torch::Tensor idx, int64_t feature_dim,
             torch::Tensor cache, torch::Tensor cache_table)
{

    Timer readTimer;
    // open file
    int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);

    int64_t feature_size = feature_dim * sizeof(float);

    int64_t num_idx = idx.numel();

    // float *read_buffer = (float *)aligned_alloc(
    //     ALIGNMENT, ALIGNMENT * 2 * atoi(getenv("GINEX_NUM_THREADS")));
    float *read_buffer =
        (float *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * num_idx);
    float *result_buffer =
        (float *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

    auto idx_data = idx.data_ptr<int64_t>();
    auto cache_data = cache.data_ptr<float>();
    auto cache_table_data = cache_table.data_ptr<int32_t>();

    readTimer.reset();
    int64_t not_in_cache = 0;
    // stage 1: dispatch read requests
#pragma omp parallel for num_threads(atoi(getenv("GINEX_NUM_THREADS")))        \
    shared(not_in_cache)
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t cache_entry;
        int64_t read_size;

        i = idx_data[n];
        cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            continue;
        } else {
#pragma omp atomic
            not_in_cache++;
            offset = i * feature_size;
            aligned_offset = offset & (long)~(ALIGNMENT - 1);
            residual = offset - aligned_offset;

            if (residual + feature_size > ALIGNMENT) {
                read_size = ALIGNMENT * 2;
            } else {
                read_size = ALIGNMENT;
            }
            if (pread(feature_fd,
                      read_buffer + (ALIGNMENT * 2 * n) / sizeof(float),
                      read_size, aligned_offset) == -1) {
                fprintf(stderr, "ERROR: %s\n", strerror(errno));
            }
        }
    }
    auto floating = readTimer.elapsed_and_reset() / 1000.0;

    std::cout << "prepare read cost " << floating << " s" << std::endl;
    // stage 2: collect the data and copy
#pragma omp parallel for num_threads(atoi(getenv("GINEX_NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t cache_entry;
        int64_t read_size;

        i = idx_data[n];
        cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            memcpy(result_buffer + feature_dim * n,
                   cache_data + cache_entry * feature_dim, feature_size);
        } else {
            offset = i * feature_size;
            aligned_offset = offset & (long)~(ALIGNMENT - 1);
            residual = offset - aligned_offset;

            if (residual + feature_size > ALIGNMENT) {
                read_size = ALIGNMENT * 2;
            } else {
                read_size = ALIGNMENT;
            }
            memcpy(result_buffer + feature_dim * n,
                   read_buffer + (ALIGNMENT * 2 * n + residual) / sizeof(float),
                   feature_size);
        }
    }

    floating = readTimer.elapsed_and_reset() / 1000.0;
    std::cout << "memcpy copy cost " << floating << " s" << std::endl;
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result =
        torch::from_blob(result_buffer, {num_idx, feature_dim}, options);
    free(read_buffer);
    close(feature_fd);

    return std::make_tuple(result, not_in_cache);
}

torch::Tensor gather_ginex_async(std::string feature_file, torch::Tensor idx,
                                 int64_t feature_dim, torch::Tensor cache,
                                 torch::Tensor cache_table)
{
    Timer readTimer;
    // open file

    int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);
    lseek(feature_fd, 0, SEEK_SET);  // reset the point

    int64_t feature_size = feature_dim * sizeof(float);

    int64_t num_idx = idx.numel();

    struct aiocb *wrlist = new struct aiocb[num_idx];

    // float *read_buffer = (float *)aligned_alloc(
    //     ALIGNMENT, ALIGNMENT * 2 * atoi(getenv("GINEX_NUM_THREADS")));
    float *read_buffer =
        (float *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * num_idx);
    float *result_buffer =
        (float *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

    auto idx_data = idx.data_ptr<int64_t>();
    auto cache_data = cache.data_ptr<float>();
    auto cache_table_data = cache_table.data_ptr<int32_t>();

    readTimer.reset();
    // stage 1: dispatch read requests
#pragma omp parallel for num_threads(128)
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t cache_entry;
        int64_t read_size;

        i = idx_data[n];
        cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            continue;
        } else {
            offset = i * feature_size;
            aligned_offset = offset & (long)~(ALIGNMENT - 1);
            residual = offset - aligned_offset;
            if (residual + feature_size > ALIGNMENT) {
                read_size = ALIGNMENT * 2;
            } else {
                read_size = ALIGNMENT;
            }

            wrlist[n].aio_fildes = feature_fd;
            wrlist[n].aio_buf =
                read_buffer + (ALIGNMENT * 2 * n) / sizeof(float);
            wrlist[n].aio_offset = aligned_offset;
            wrlist[n].aio_nbytes = read_size;

            if (aio_read(&wrlist[n]) == -1) {
                fprintf(stderr, "ERROR: %s, index: %d\n", strerror(errno), n);
            }
        }
    }

    auto floating = readTimer.elapsed_and_reset() / 1000.0;
    std::cout << "prepare read cost " << floating << " s" << std::endl;
    // stage 2: collect the data and copy
#pragma omp parallel for num_threads(128)
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t cache_entry;
        int64_t read_size;

        i = idx_data[n];
        cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            memcpy(result_buffer + feature_dim * n,
                   cache_data + cache_entry * feature_dim, feature_size);
        } else {
            offset = i * feature_size;
            aligned_offset = offset & (long)~(ALIGNMENT - 1);
            residual = offset - aligned_offset;

            if (residual + feature_size > ALIGNMENT) {
                read_size = ALIGNMENT * 2;
            } else {
                read_size = ALIGNMENT;
            }

            while (aio_error(&wrlist[n]) == EINPROGRESS) {}

            if (aio_return(&wrlist[n]) < 0) {
                fprintf(stderr, "index: %d, return error\n", n);
            }
            memcpy(result_buffer + feature_dim * n,
                   read_buffer + (ALIGNMENT * 2 * n + residual) / sizeof(float),
                   feature_size);
        }
    }

    floating = readTimer.elapsed_and_reset() / 1000.0;
    std::cout << "memcpy copy cost " << floating << " s" << std::endl;
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result =
        torch::from_blob(result_buffer, {num_idx, feature_dim}, options);
    free(read_buffer);
    close(feature_fd);
    delete (wrlist);

    return result;
}

int64_t tensor_size(torch::Tensor transfer_tensor, int element_size)
{
    // Returns the size of tensor
    int dim = transfer_tensor.dim();
    int64_t num_element = 1;
    for (int dx = 0; dx < dim; ++dx) {
        num_element *= transfer_tensor.sizes()[dx];
    }
    return (int64_t)num_element * element_size;
}

torch::Tensor alloc_uvm_indice(int64_t num_nodes)
{
    // initial a uvm-access indice
    int32_t *feature_indice;
    gpuErrorcheck(
        cudaMallocManaged(&feature_indice, num_nodes * sizeof(int32_t)));
    gpuErrorcheck(cudaMemset(feature_indice, -1, num_nodes * sizeof(int32_t)));

    auto options = torch::TensorOptions()
                       .dtype(torch::kInt32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result = torch::from_blob(feature_indice, {num_nodes, 1}, options);
    return result;
}

torch::Tensor alloc_uvm_cache(std::string feature_file, torch::Tensor indice,
                              int feature_dim)
{
    int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);
    if (feature_fd < 0) {
        log_error("open file.. check if the feature path exists.. ");
    }
    int num_idx = indice.numel();
    int64_t feature_size = feature_dim * sizeof(float);

    float *feature_cache;
    gpuErrorcheck(cudaMallocManaged(&feature_cache, num_idx * feature_size));
    log_info("alloc uvm cache finish..");
    float *read_buffer =
        (float *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * num_idx);
#pragma omp parallel for num_threads(atoi(getenv("GINEX_NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t read_size;

        int64_t i = idx_data[n];
        offset = i * feature_size;
        aligned_offset = offset & (long)~(ALIGNMENT - 1);
        residual = offset - aligned_offset;

        if (residual + feature_size > ALIGNMENT) {
            read_size = ALIGNMENT * 2;
        } else {
            read_size = ALIGNMENT;
        }
        memcpy(feature_cache + feature_dim * n,
                read_buffer + (ALIGNMENT * 2 * n + residual) / sizeof(float),
                feature_size);
    }
    log_info("filling cache finish..");
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result =
        torch::from_blob(feature_cache, {num_idx, feature_dim}, options);
    free(read_buffer);
    close(feature_fd);
    return result;
}

PYBIND11_MODULE(gather, m)
{
    m.def("gather_ginex", &gather_ginex, "gather for ginex",
          py::call_guard<py::gil_scoped_release>());
    m.def("gather_ginex_async", &gather_ginex_async,
          "gather for ginex in async",
          py::call_guard<py::gil_scoped_release>());
    m.def("gather_mmap", &gather_mmap, "gather for PyG+åå",
          py::call_guard<py::gil_scoped_release>());
    m.def("alloc_uvm_indice", &alloc_uvm_indice,
          "alloc memory for uvm indice access",
          py::call_guard<py::gil_scoped_release>());
    m.def("alloc_uvm_cache", &alloc_uvm_cache,
          "alloc memory for uvm cache access",
          py::call_guard<py::gil_scoped_release>());
    m.def("get_size_in_bytes", &tensor_size, "get tensor size in bytes",
          py::call_guard<py::gil_scoped_release>());
}
