#include <ATen/ATen.h>
#include <Python.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <pthread.h>
#include <unistd.h>
#include <libaio.h>
#include <sys/syscall.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <vector>
#include <cmath>
#define ALIGNMENT 4096

/*
    syscall for async read/write
*/
// int io_setup(unsigned nr, aio_context_t *ctxp) {
// 	return syscall(__NR_io_setup, nr, ctxp);
// }

// int io_destroy(aio_context_t ctx) {
// 	return syscall(__NR_io_destroy, ctx);
// }

// int io_submit(aio_context_t ctx, long nr, struct iocb **iocbpp) {
// 	return syscall(__NR_io_submit, ctx, nr, iocbpp);
// }

// int io_getevents(aio_context_t ctx, long min_nr, long max_nr, struct io_event *events, struct timespec *timeout) {
// 	return syscall(__NR_io_getevents, ctx, min_nr, max_nr, events, timeout);
// }

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
    // shared(not_in_cache)
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

        i = idx_data[n];
        cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            memcpy(result_buffer + feature_dim * n,
                   cache_data + cache_entry * feature_dim, feature_size);
        } else {
            offset = i * feature_size;
            aligned_offset = offset & (long)~(ALIGNMENT - 1);
            residual = offset - aligned_offset;
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

std::tuple<torch::Tensor, int64_t> 
gather_ginex_async(std::string feature_file, torch::Tensor idx,
                                 int64_t feature_dim, torch::Tensor cache,
                                 torch::Tensor cache_table)
{
    Timer readTimer;
    // open file

    int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);
    if (feature_fd < 0) {
        std::cout << "file open failed\n";
        return std::make_tuple(torch::Tensor(), 0);
    }
    lseek(feature_fd, 0, SEEK_SET);  // reset the point

    int64_t feature_size = feature_dim * sizeof(float);

    int64_t num_idx = idx.numel();
    int64_t not_in_cache = 0;

    io_context_t ctx;
    int ret;
    // float *read_buffer = (float *)aligned_alloc(
    //     ALIGNMENT, ALIGNMENT * 2 * atoi(getenv("GINEX_NUM_THREADS")));
    float *result_buffer =
        (float *)aligned_alloc(ALIGNMENT, feature_size * num_idx);
    int64_t *indice = (int64_t *)malloc(num_idx * sizeof(int64_t));
    int64_t *prefix = (int64_t *)malloc(num_idx * sizeof(int64_t));

    memset(&ctx, 0, sizeof(io_context_t));
    memset(indice, 0, num_idx * sizeof(int64_t));
    memset(prefix, 0, num_idx * sizeof(int64_t));

    auto idx_data = idx.data_ptr<int64_t>();
    auto cache_data = cache.data_ptr<float>();
    auto cache_table_data = cache_table.data_ptr<int32_t>();


    readTimer.reset();

#pragma omp parallel for num_threads(128)
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i = idx_data[n];
        int64_t cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            continue;
        } else {
#pragma omp atomic
            not_in_cache += 1;
            indice[n] = 1;
        }
    }
    int submit_batch = 512;

    ret = io_setup(submit_batch, &ctx);
    if (ret != 0) {
        printf("io_setup error: %d :%s\n", ret, strerror(-ret));  
    } else {
        std::cout << "after io_setup, the ctx is " << ctx << std::endl;
    }
    std::cout << "In async mode, the entry not in cache is " << not_in_cache << std::endl;

    float *read_buffer =
        (float *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * not_in_cache);
    
    for (int i = 1; i < num_idx; ++ i) {
        prefix[i] = prefix[i - 1] + indice[i - 1];
    }
    printf("prefix %ld is %ld, not_in_cache is %ld\n", num_idx, prefix[num_idx - 1], not_in_cache);

    // stage 1: dispatch read requests
    std::vector<iocb*> request_stack;
    struct iocb *io_req = new iocb[not_in_cache];
    memset(io_req, 0, sizeof(iocb) * not_in_cache);

#pragma omp parallel for num_threads(128)
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t cache_entry;
        int64_t read_size;
        int64_t residual;

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
            io_prep_pread(&io_req[prefix[n]], feature_fd, (void *)(read_buffer + (ALIGNMENT * 2 * prefix[n]) / sizeof(float)), read_size, aligned_offset);
        }
    }
    printf("prepare read finish\n");
    int done_task = 0;
    while (done_task < not_in_cache) {
        request_stack.clear();
        int submit_task = std::min(submit_batch, int(not_in_cache - done_task));
        request_stack.resize(submit_task);
        for (int i = 0; i < submit_task; i ++) {
            request_stack[i] = &io_req[i + done_task];
        }
        ret = io_submit(ctx, request_stack.size(), reinterpret_cast<iocb**>(request_stack.data()));
        if (ret != request_stack.size()) {
            if (ret < 0) 
                printf("io_submit: %d : %s\n", ret, strerror(-ret));
            else
                printf("could not sumbit IOs, submit number: %d, total: %d\n", ret, submit_task);
            free(read_buffer);
            free(result_buffer);
            free(indice);
            free(prefix);
            request_stack.clear();
            close(feature_fd);
            delete(io_req);
            io_destroy(ctx);
            return std::make_tuple(torch::Tensor(), 0);
        }
        // stage 2: collect the data and copy
        struct io_event *events = new io_event[submit_task];
        ret = io_getevents(ctx, submit_task, submit_task, events, NULL);
        if (ret != request_stack.size()) {
            printf("io_submit: %d : %s\n", ret, strerror(-ret));
        }
        delete(events);
        done_task += submit_task;
    }
    auto floating = readTimer.elapsed_and_reset() / 1000.0;
    printf("transfer cost %.2f s\n", floating);
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
            memcpy(result_buffer + feature_dim * n,
                   read_buffer + (ALIGNMENT * 2 * prefix[n] + residual) / sizeof(float),
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
    free(result_buffer);
    free(indice);
    free(prefix);

    close(feature_fd);
    delete(io_req);
    request_stack.clear();
    io_destroy(ctx);
    return std::make_tuple(result, not_in_cache);
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


PYBIND11_MODULE(gather, m)
{
    m.def("gather_ginex", &gather_ginex, "gather for ginex",
          py::call_guard<py::gil_scoped_release>());
    m.def("gather_ginex_async", &gather_ginex_async,
          "gather for ginex in async",
          py::call_guard<py::gil_scoped_release>());
    m.def("gather_mmap", &gather_mmap, "gather for PyG+",
          py::call_guard<py::gil_scoped_release>());
    m.def("get_size_in_bytes", &tensor_size, "get tensor size in bytes",
          py::call_guard<py::gil_scoped_release>());
}
