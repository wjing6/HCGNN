#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include <iostream>
#include <thrust/device_vector.h>

#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <quiver/common.hpp>
#include <quiver/functor.cu.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/reindex.cu.hpp>
#include <quiver/stream_pool.hpp>
#include <quiver/trace.hpp>
#include <quiver/zip.hpp>
#include <thrust/remove.h>

template <typename IdType>
HostOrderedHashTable<IdType> *
FillWithDuplicates(const IdType *const input, const size_t num_input,
                   cudaStream_t stream,
                   thrust::device_vector<IdType> &unique_items)
{
    const auto policy = thrust::cuda::par.on(stream);
    const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

    const dim3 grid(num_tiles);
    const dim3 block(BLOCK_SIZE);

    auto host_table = new HostOrderedHashTable<IdType>(num_input, 1);
    DeviceOrderedHashTable<IdType> device_table = host_table->DeviceHandle();

    generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE>
        <<<grid, block, 0, stream>>>(input, num_input, device_table);
    thrust::device_vector<int> item_prefix(num_input + 1, 0);

    using it = thrust::counting_iterator<IdType>;
    using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;
    thrust::for_each(it(0), it(num_input),
                     [count = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table,
                      in = input] __device__(IdType i) mutable {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i) { count[i] = 1; }
                     });
    thrust::exclusive_scan(item_prefix.begin(), item_prefix.end(),
                           item_prefix.begin());
    size_t tot = item_prefix[num_input];
    unique_items.resize(tot);

    thrust::for_each(it(0), it(num_input),
                     [prefix = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table, in = input,
                      u = thrust::raw_pointer_cast(
                          unique_items.data())] __device__(IdType i) mutable {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i) {
                             mapping.local = prefix[i];
                             u[prefix[i]] = in[i];
                         }
                     });
    return host_table;
}

namespace quiver
{
template <typename T>
void replicate_fill(size_t n, const T *counts, const T *values, T *outputs)
{
    for (size_t i = 0; i < n; ++i) {
        const size_t c = counts[i];
        std::fill(outputs, outputs + c, values[i]);
        outputs += c;
    }
}

class TorchQuiver
{
    using torch_quiver_t = quiver<int64_t, CUDA>;
    torch_quiver_t quiver_;
    stream_pool pool_;

  public:
    TorchQuiver(torch_quiver_t quiver, int device = 0, int num_workers = 4)
        : quiver_(std::move(quiver))
    {
        pool_ = stream_pool(num_workers);
    }

    using T = int64_t;
    using W = float;

    // deprecated, not compatible with AliGraph
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub(const torch::Tensor &vertices, const torch::Tensor &cache_idx, int k) const
    {
        return sample_sub_with_stream(0, vertices, cache_idx, k);
    }

    void cal_neighbor_prob(int stream_num, const torch::Tensor last_prob,
                           torch::Tensor cur_prob, int k)
    {
        cudaStream_t stream = 0;
        auto p = quiver_.csr();
        const T *indptr = p.first;
        const T *indices = p.second;
        size_t block = (last_prob.size(0) + 127) / 128;
        cal_next<<<block, 128, 0, stream>>>(
            last_prob.data_ptr<float>(), cur_prob.data_ptr<float>(),
            cur_prob.size(0), k, indptr, indices);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    sample_neighbor(int stream_num, const torch::Tensor &vertices, const torch::Tensor &cache_idx, int k)
    {
        // according the embedding indice, filter the nodes
        cudaStream_t stream = 0;
        if (!pool_.empty()) { stream = (pool_)[stream_num]; }
        const auto policy = thrust::cuda::par.on(stream);
        const size_t bs = vertices.size(0);
        thrust::device_vector<T> inputs;
        thrust::device_vector<T> outputs;
        thrust::device_vector<T> output_counts;
        sample_kernel(stream, vertices, cache_idx, k, inputs, outputs, output_counts);
        torch::Tensor neighbors =
            torch::empty(outputs.size(), vertices.options());
        torch::Tensor counts =
            torch::empty(vertices.size(0), vertices.options());
        thrust::copy(outputs.begin(), outputs.end(), neighbors.data_ptr<T>());
        thrust::copy(output_counts.begin(), output_counts.end(),
                     counts.data_ptr<T>());
        return std::make_tuple(neighbors, counts);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    sample_kernel(const cudaStream_t stream, const torch::Tensor &vertices,
                  const torch::Tensor &cache_idx,
                  int k, thrust::device_vector<T> &inputs,
                  thrust::device_vector<T> &outputs,
                  thrust::device_vector<T> &output_counts) const
    {
        // what is output? what is output_counts?
        // outputs: the global ID of sampled node, output_counts: the row ptr
        T tot = 0;
        const auto policy = thrust::cuda::par.on(stream);
        thrust::device_vector<T> output_ptr;
        thrust::device_vector<T> output_idx;
        thrust::device_vector<T> cache_idx_vec;

        // thrust::device_vector<T> test_output;

        const T *p = vertices.data_ptr<T>();
        const T *idx_ptr = cache_idx.data_ptr<T>();

        const size_t bs = vertices.size(0);
        const size_t num_nodes = cache_idx.size(0);
        {
            TRACE_SCOPE("alloc_1");
            inputs.resize(bs);
            output_counts.resize(bs);
            output_ptr.resize(bs);
            cache_idx_vec.resize(num_nodes);

            // test_output.resize(bs);
        }
        // output_ptr is exclusive prefix sum of output_counts(neighbor counts
        // <= k)
        {
            TRACE_SCOPE("prepare");
            thrust::copy(p, p + bs, inputs.begin());
            thrust::copy(idx_ptr, idx_ptr + num_nodes, cache_idx_vec.begin());
            // quiver_.to_local(stream, inputs);
            quiver_.degree_with_stale(stream, inputs.data(), inputs.data() + inputs.size(),
                            thrust::raw_pointer_cast(cache_idx_vec.data()), output_counts.data());
            // quiver_.degree(stream, inputs.data(), inputs.data() + inputs.size(), test_output.data());
            if (k >= 0) {
                thrust::transform(policy, output_counts.begin(),
                                  output_counts.end(), output_counts.begin(),
                                  cap_by<T>(k));
                // thrust::transform(policy, test_output.begin(),
                //                   test_output.end(), test_output.begin(),
                //                   cap_by<T>(k));
            }
            thrust::exclusive_scan(policy, output_counts.begin(),
                                   output_counts.end(), output_ptr.begin());
            tot = thrust::reduce(policy, output_counts.begin(),
                                 output_counts.end());
            // T test_tot = thrust::reduce(policy, test_output.begin(),
            //                      test_output.end());
	        // std::cout << "before cache, number is " << test_tot<< " ;after cache, total number is " << tot << std::endl;
        }
        {
            TRACE_SCOPE("alloc_2");
            outputs.resize(tot);
            output_idx.resize(tot);
        }
        // outputs[outptr[i], outptr[i + 1]) are unique neighbors of inputs[i]
        // {
        //     TRACE_SCOPE("sample");
        //     quiver_.sample(stream, inputs.begin(), inputs.end(),
        //                    output_ptr.begin(), output_counts.begin(),
        //                    outputs.data(), output_eid.data());
        // }
        {
            TRACE_SCOPE("sample");
            quiver_.new_sample( 
                stream, k, thrust::raw_pointer_cast(inputs.data()),
                thrust::raw_pointer_cast(cache_idx_vec.data()),
                inputs.size(), thrust::raw_pointer_cast(output_ptr.data()),
                thrust::raw_pointer_cast(output_counts.data()),
                thrust::raw_pointer_cast(outputs.data()),
                thrust::raw_pointer_cast(output_idx.data()));
        }
        torch::Tensor out_neighbor;
        torch::Tensor out_eid;


        return std::make_tuple(out_neighbor, out_eid);
    }

    static void reindex_kernel(const cudaStream_t stream,
                               thrust::device_vector<T> &inputs, 
                               thrust::device_vector<T> &outputs,
                               thrust::device_vector<T> &subset)
    {
        const auto policy = thrust::cuda::par.on(stream);
        HostOrderedHashTable<T> *table;
        // reindex
        {
            {
                TRACE_SCOPE("reindex 0");
                subset.resize(inputs.size() + outputs.size());
                thrust::copy(policy, inputs.begin(), inputs.end(),
                             subset.begin());
                thrust::copy(policy, outputs.begin(), outputs.end(),
                             subset.begin() + inputs.size());
                thrust::device_vector<T> unique_items;
                unique_items.clear();
                table =
                    FillWithDuplicates(thrust::raw_pointer_cast(subset.data()),
                                       subset.size(), stream, unique_items);
                subset.resize(unique_items.size());
                thrust::copy(policy, unique_items.begin(), unique_items.end(),
                             subset.begin());
            }
            {
                TRACE_SCOPE("permute");
                // thrust::device_vector<T> s1;
                // s1.reserve(subset.size());
                // _reindex_with(policy, inputs, subset, s1);
                // complete_permutation(s1, subset.size(), stream);
                // subset = permute(s1, subset, stream);

                // thrust::device_vector<T> s2;
                // inverse_permutation(s1, s2, stream);
                // permute_value(s2, outputs, stream);
                DeviceOrderedHashTable<T> device_table = table->DeviceHandle();
                thrust::for_each(
                    policy, outputs.begin(), outputs.end(),
                    [device_table] __device__(T & id) mutable {
                        using Iterator =
                            typename DeviceOrderedHashTable<T>::Iterator;
                        Iterator iter = device_table.Search(id);
                        id = static_cast<T>((*iter).local);
                    });
            }
            delete table;
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub_with_stream(int stream_num, const torch::Tensor &vertices,
                           const torch::Tensor &cache_idx,
                           int k) const
    {
        TRACE_SCOPE(__func__);
        cudaStream_t stream = 0;
        if (!pool_.empty()) { stream = (pool_)[stream_num]; }
        const auto policy = thrust::cuda::par.on(stream);
        thrust::device_vector<T> inputs;
        thrust::device_vector<T> outputs;
        thrust::device_vector<T> output_counts;
        thrust::device_vector<T> subset;
        sample_kernel(stream, vertices, cache_idx, k, inputs, outputs, output_counts);
        int tot = outputs.size();

        reindex_kernel(stream, inputs, outputs, subset);

        torch::Tensor out_vertices =
            torch::empty(subset.size(), vertices.options());
        torch::Tensor row_idx = torch::empty(tot, vertices.options());
        torch::Tensor col_idx = torch::empty(tot, vertices.options());
        {
            TRACE_SCOPE("prepare output");
            thrust::device_vector<T> prefix_count(output_counts.size());
            thrust::device_vector<T> seq(output_counts.size());
            thrust::sequence(policy, seq.begin(), seq.end());
            thrust::exclusive_scan(policy, output_counts.begin(),
                                   output_counts.end(), prefix_count.begin());

            const size_t m = inputs.size();
            using it = thrust::counting_iterator<T>;
            thrust::for_each(
                policy, it(0), it(m),
                [prefix = thrust::raw_pointer_cast(prefix_count.data()),
                 count = thrust::raw_pointer_cast(output_counts.data()),
                 in = thrust::raw_pointer_cast(seq.data()),
                 out = thrust::raw_pointer_cast(
                     row_idx.data_ptr<T>())] __device__(T i) {
                    for (int j = 0; j < count[i]; j++) {
                        out[prefix[i] + j] = in[i];
                    }
                });
            thrust::copy(subset.begin(), subset.end(),
                         out_vertices.data_ptr<T>());
            thrust::copy(outputs.begin(), outputs.end(), col_idx.data_ptr<T>());
        }
        return std::make_tuple(out_vertices, row_idx, col_idx);
    }
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    reindex_single(torch::Tensor inputs, torch::Tensor outputs, torch::Tensor count)
    {
        using T = int64_t;
        cudaStream_t stream = 0;
        const auto policy = thrust::cuda::par.on(stream);
        thrust::device_vector<T> total_inputs(inputs.size(0));
        thrust::device_vector<T> total_outputs(outputs.size(0));
        thrust::device_vector<T> input_prefix(inputs.size(0));
        const T *ptr;
        size_t bs;
        ptr = count.data_ptr<T>();
        bs = inputs.size(0);
        thrust::copy(ptr, ptr + bs, input_prefix.begin());
        ptr = inputs.data_ptr<T>();
        thrust::copy(ptr, ptr + bs, total_inputs.begin());
        thrust::exclusive_scan(policy, input_prefix.begin(), input_prefix.end(),
                               input_prefix.begin());
        ptr = outputs.data_ptr<T>();
        bs = outputs.size(0);
        thrust::copy(ptr, ptr + bs, total_outputs.begin());

        const size_t m = inputs.size(0);
        using it = thrust::counting_iterator<T>;

        thrust::device_vector<T> subset;
        TorchQuiver::reindex_kernel(stream, total_inputs, total_outputs,
                                    subset);
        // subset 保留去重后的 global id, total_outputs 则是对应重新map的 local id
        int tot = total_outputs.size();
        torch::Tensor out_vertices =
            torch::empty(subset.size(), inputs.options());
        torch::Tensor row_idx = torch::empty(tot, inputs.options());
        torch::Tensor col_idx = torch::empty(tot, inputs.options());
        {
            thrust::device_vector<T> seq(count.size(0));
            thrust::sequence(policy, seq.begin(), seq.end());

            thrust::for_each(
                policy, it(0), it(m),
                [prefix = thrust::raw_pointer_cast(input_prefix.data()),
                 count = count.data_ptr<T>(),
                 in = thrust::raw_pointer_cast(seq.data()),
                 out = thrust::raw_pointer_cast(
                     row_idx.data_ptr<T>())] __device__(T i) {
                    for (int j = 0; j < count[i]; j++) {
                        out[prefix[i] + j] = in[i];
                    }
                });
            thrust::copy(subset.begin(), subset.end(),
                         out_vertices.data_ptr<T>());
            thrust::copy(total_outputs.begin(), total_outputs.end(),
                         col_idx.data_ptr<T>());
        }
        return std::make_tuple(out_vertices, row_idx, col_idx);
    }
};

TorchQuiver new_quiver_from_csr_array(torch::Tensor &input_indptr,
                                      torch::Tensor &input_indices,
                                      torch::Tensor &input_edge_idx,
                                      int device = 0, bool cuda = false)
{

    cudaSetDevice(device);
    TRACE_SCOPE(__func__);

    using T = typename TorchQuiver::T;

    check_eq<int64_t>(input_indptr.dim(), 1);
    const size_t node_count = input_indptr.size(0);

    check_eq<int64_t>(input_indices.dim(), 1);
    const size_t edge_count = input_indices.size(0);

    bool use_eid = input_edge_idx.size(0) == edge_count;

    /*
    In Zero-Copy Mode, We Do These Steps:
    0. Copy The Data If Needed
    1. Register Buffer As Mapped Pinned Memory
    2. Get Device Pointer In GPU Memory Space
    3. Intiliaze A Quiver Instance And Return
    */

    T *indptr_device_pointer = nullptr;
    T *indices_device_pointer = nullptr;
    T *edge_id_device_pointer = nullptr;
    {/*if (!cuda) {
         const T *indptr_original = reinterpret_cast<const T
     *>(input_indptr.data_ptr<T>());
         // Register Buffer As Mapped Pinned Memory
         quiverRegister((void *)indptr_original, sizeof(T) * node_count,
                          cudaHostRegisterMapped);
         // Get Device Pointer In GPU Memory Space
         cudaHostGetDevicePointer((void **)&indptr_device_pointer,
                                  (void *)indptr_original, 0);
     } else */
     {const T *indptr_original =
          reinterpret_cast<const T *>(input_indptr.data_ptr<T>());
    T *indptr_copy;
    cudaMalloc((void **)&indptr_copy, sizeof(T) * node_count);
    cudaMemcpy((void *)indptr_copy, (void *)indptr_original,
               sizeof(T) * node_count, cudaMemcpyDefault);
    indptr_device_pointer = indptr_copy;
}

}  // namespace quiver
// std::cout<<"mapped indptr"<<std::endl;
{
    if (!cuda) {
        const T *indices_original =
            reinterpret_cast<const T *>(input_indices.data_ptr<T>());
        // Register Buffer As Mapped Pinned Memory
        quiverRegister((void *)indices_original, sizeof(T) * edge_count,
                       cudaHostRegisterMapped);
        // Get Device Pointer In GPU Memory Space
        cudaHostGetDevicePointer((void **)&indices_device_pointer,
                                 (void *)indices_original, 0);
    } else {
        const T *indices_original =
            reinterpret_cast<const T *>(input_indices.data_ptr<T>());
        T *indices_copy;
        cudaMalloc((void **)&indices_copy, sizeof(T) * edge_count);
        cudaMemcpy((void *)indices_copy, (void *)indices_original,
                   sizeof(T) * edge_count, cudaMemcpyDefault);
        indices_device_pointer = indices_copy;
    }
}

// std::cout<<"mapped indices"<<std::endl;
if (use_eid) {
    if (!cuda) {
        const T *id_original =
            reinterpret_cast<const T *>(input_edge_idx.data_ptr<T>());
        // Register Buffer As Mapped Pinned Memory
        quiverRegister((void *)id_original, sizeof(T) * edge_count,
                       cudaHostRegisterMapped);
        // Get Device Pointer In GPU Memory Space
        cudaHostGetDevicePointer((void **)&edge_id_device_pointer,
                                 (void *)id_original, 0);
    } else {
        const T *id_original =
            reinterpret_cast<const T *>(input_edge_idx.data_ptr<T>());
        T *id_copy;
        cudaMalloc((void **)&id_copy, sizeof(T) * edge_count);
        cudaMemcpy((void *)id_copy, (void *)id_original, sizeof(T) * edge_count,
                   cudaMemcpyDefault);
        edge_id_device_pointer = id_copy;
    }
}

// std::cout<<"mapped edge id "<<std::endl;
// initialize Quiver instance
using Q = quiver<int64_t, CUDA>;
Q quiver = Q::New(indptr_device_pointer, indices_device_pointer,
                  edge_id_device_pointer, node_count - 1, edge_count);
return TorchQuiver(std::move(quiver), device);
}

TorchQuiver new_quiver_from_edge_index(size_t n,
                                       py::array_t<int64_t> &input_edges,
                                       py::array_t<int64_t> &input_edge_idx,
                                       int device = 0)
{
    cudaSetDevice(device);
    TRACE_SCOPE(__func__);
    using T = typename TorchQuiver::T;
    py::buffer_info edges = input_edges.request();
    py::buffer_info edge_idx = input_edge_idx.request();
    check_eq<int64_t>(edges.ndim, 2);
    check_eq<int64_t>(edges.shape[0], 2);
    const size_t m = edges.shape[1];
    check_eq<int64_t>(edge_idx.ndim, 1);

    bool use_eid = edge_idx.shape[0] == m;

    thrust::device_vector<T> row_idx(m);
    thrust::device_vector<T> col_idx(m);
    {
        const T *p = reinterpret_cast<const T *>(edges.ptr);
        thrust::copy(p, p + m, row_idx.begin());
        thrust::copy(p + m, p + m * 2, col_idx.begin());
    }
    thrust::device_vector<T> edge_idx_;
    if (use_eid) {
        edge_idx_.resize(m);
        const T *p = reinterpret_cast<const T *>(edge_idx.ptr);
        thrust::copy(p, p + m, edge_idx_.begin());
    }
    using Q = quiver<int64_t, CUDA>;
    Q quiver = Q::New(static_cast<T>(n), std::move(row_idx), std::move(col_idx),
                      std::move(edge_idx_));
    return TorchQuiver(std::move(quiver), device);
}
}  // namespace quiver

void register_cuda_quiver_sample(pybind11::module &m)
{
    m.def("device_quiver_from_edge_index", &quiver::new_quiver_from_edge_index);
    m.def("device_quiver_from_csr_array", &quiver::new_quiver_from_csr_array);
    // py::gil_scoped_release release the python GIL!
    py::class_<quiver::TorchQuiver>(m, "Quiver")
        .def("sample_sub", &quiver::TorchQuiver::sample_sub_with_stream,
             py::call_guard<py::gil_scoped_release>())
        .def("sample_neighbor", &quiver::TorchQuiver::sample_neighbor,
             py::call_guard<py::gil_scoped_release>())
        .def("cal_neighbor_prob", &quiver::TorchQuiver::cal_neighbor_prob,
             py::call_guard<py::gil_scoped_release>())
        .def("reindex_single", &quiver::TorchQuiver::reindex_single,
             py::call_guard<py::gil_scoped_release>());
}
