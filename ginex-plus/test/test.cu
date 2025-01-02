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
#include <omp.h>
#include <pthread.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <unistd.h>

#include "../utils/log/log.h"

#define ALIGNMENT 4096

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
}


void test_direct_copy(std::string feature_file, torch::Tensor idx,
                      int64_t total_idx, int64_t feature_dim)
{
    int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);
    int64_t feature_size = feature_dim * sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();

    float *feature_buffer;
    log_info("begin loading...");
    // Timer timer;
    // cout << "loading start" << endl;
    checkCuda(cudaMallocHost((void **)&feature_buffer, feature_size * total_idx));
    // feature_buffer = (float *)malloc(feature_size * total_idx);
    read(feature_fd, feature_buffer, feature_size * total_idx);
    log_info("finish loading...");

    float *read_buffer = (float *)aligned_alloc(
        ALIGNMENT, ALIGNMENT * 2 * atoi(getenv("TEST_NUM_THREADS")));
    float *result_buffer =
        (float *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

    auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for num_threads(atoi(getenv("TEST_NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i = idx_data[n];
        memcpy(result_buffer + feature_dim * n,
               feature_buffer + i * feature_dim, feature_size);
    }
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result =
        torch::from_blob(result_buffer, {num_idx, feature_dim}, options);
    free(read_buffer);
    close(feature_fd);
    log_info("finish");

    return;
}

void test_read_and_copy(std::string feature_file, torch::Tensor idx,
                        int64_t feature_dim)
{
    int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);

    int64_t feature_size = feature_dim * sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();
    log_info("register file, begin reading and copying...");
    float *read_buffer = (float *)aligned_alloc(
        ALIGNMENT, ALIGNMENT * 2 * atoi(getenv("TEST_NUM_THREADS")));
    float *result_buffer =
        (float *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

    auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for num_threads(atoi(getenv("TEST_NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t read_size;

        {
            offset = i * feature_size;
            aligned_offset = offset & (long)~(ALIGNMENT - 1);
            residual = offset - aligned_offset;

            if (residual + feature_size > ALIGNMENT) {
                read_size = ALIGNMENT * 2;
            } else {
                read_size = ALIGNMENT;
            }
            if (pread(feature_fd,
                      read_buffer + (ALIGNMENT * 2 * omp_get_thread_num()) /
                                        sizeof(float),
                      read_size, aligned_offset) == -1) {
                fprintf(stderr, "ERROR: %s\n", strerror(errno));
            }
            memcpy(result_buffer + feature_dim * n,
                   read_buffer +
                       (ALIGNMENT * 2 * omp_get_thread_num() + residual) /
                           sizeof(float),
                   feature_size);
        }
    }

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result =
        torch::from_blob(result_buffer, {num_idx, feature_dim}, options);
    free(read_buffer);
    close(feature_fd);

    log_info("finish");
    return;
}


PYBIND11_MODULE(test, m)
{
    m.def("test_direct_copy", &test_direct_copy, "test copy in CPU",
          py::call_guard<py::gil_scoped_release>());
    m.def("test_read_and_copy", &test_read_and_copy, "test read and copy",
          py::call_guard<py::gil_scoped_release>());
}
