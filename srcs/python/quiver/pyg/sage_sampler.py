from tkinter import E
import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch_quiver as qv
from typing import List, Tuple, NamedTuple, Generic, TypeVar
from dataclasses import dataclass
import torch.multiprocessing as mp
import itertools
import time
import os
import quiver
import quiver.utils as quiver_utils


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

__all__ = ["GraphSageSampler", "MixedGraphSageSampler", "SampleJob"]


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


@dataclass(frozen=True)
class _FakeDevice(object):
    pass

@dataclass(frozen=True)
class _StopWork(object):
    pass


class GraphSageSampler:
    r"""
    Quiver's GraphSageSampler behaves just like Pyg's `NeighborSampler` but with much higher performance.
    It can work in `UVA` mode or `GPU` mode. You can set `mode=GPU` if you have enough GPU memory to place graph's topology data which will offer the best sample performance.
    When your graph is too big for GPU memory, you can set `mode=UVA` to still use GPU to perform sample but place the data in host memory. `UVA` mode suffers 30%-40% performance loss compared to `GPU` mode
    but is much faster than CPU sampling(normally 16x~20x) and it consumes much less GPU memory compared to `GPU` mode.

    Args:
        csr_topo (quiver.CSRTopo): A quiver.CSRTopo for graph topology
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to `sizes[l] = -1`, all neighbors are included
            in layer `l`.
        device (int): Device which sample kernel will be launched
        mode (str): Sample mode, choices are [`UVA`, `GPU`, `CPU`], default is `UVA`.
    """
    def __init__(self,
                 csr_topo: quiver_utils.CSRTopo,
                 num_nodes,
                 sizes: List[int],
                 exp_name,
                 trace_path,
                 embedding_cache = None,
                 # consist of multi-layer 
                 device = 0,
                 mode="UVA"):

        assert mode in ["UVA",
                        "GPU",
                        "CPU"], f"sampler mode should be one of [UVA, GPU]"
        assert device is _FakeDevice or mode == "CPU" or (device >= 0 and mode != "CPU"), f"Device setting and Mode setting not compatitive"
        

        self.sizes = sizes
        self.quiver = None
        self.csr_topo = csr_topo
        self.mode = mode
        self.num_nodes = num_nodes
        self.embedding_cache = embedding_cache
        self.save_time = 0

        if embedding_cache is not None:
            self.layers = len(embedding_cache)
        self.exp_name = exp_name
        self.trace_path = trace_path
        self.sb = 0

        self.reduce = 0
        self.after = 0


        # manually increase
        # self.cur_batch = 0
        if self.mode in ["GPU", "UVA"] and device is not _FakeDevice and  device >= 0:
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.device_quiver_from_csr_array(self.csr_topo.indptr,
                                                       self.csr_topo.indices,
                                                       edge_id, device,
                                                       self.mode != "UVA")
        elif self.mode == "CPU" and device is not _FakeDevice:
            self.quiver = qv.cpu_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices)
            device = "cpu"
        
        self.device = device
        self.ipc_handle_ = None
        self.invalid_cache = torch.full([self.num_nodes], -1, dtype=torch.int64, device=self.device)

        self.batch_count = torch.zeros(1, dtype=torch.int).share_memory_()
        self.lock = mp.Lock()

    def sample_layer(self, batch, size, embedding_cache=None):
        # sample need to exclude the stale embedding, then re-include it when passing to topper-layer
        self.lazy_init_quiver()
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        n_id = batch.to(self.device)
        size = size if size != -1 else self.csr_topo.node_count
        if embedding_cache is not None:
            if self.mode in ["GPU", "UVA"]:
                n_id, count = self.quiver.sample_neighbor(0, n_id, embedding_cache, size)
            else:
                n_id, count = self.quiver.sample_neighbor(n_id, size)
        else:
            if self.mode in ["GPU", "UVA"]:
                n_id, count = self.quiver.sample_neighbor(0, n_id, self.invalid_cache, size)
            else:
                n_id, count = self.quiver.sample_neighbor(n_id, size)
        
        return n_id, count

    def lazy_init_quiver(self):

        if self.quiver is not None:
            return

        self.device = "cpu" if self.mode == "CPU" else torch.cuda.current_device()
        
    
        if "CPU"  == self.mode:
            self.quiver = qv.cpu_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices)
        else:
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.device_quiver_from_csr_array(self.csr_topo.indptr,
                                                       self.csr_topo.indices,
                                                       edge_id, self.device,
                                                       self.mode != "UVA")

    def reindex(self, inputs, outputs, counts):
        return self.quiver.reindex_single(inputs, outputs, counts)

    def sample(self, input_nodes):
        """Sample k-hop neighbors from input_nodes

        Args:
            input_nodes (torch.LongTensor): seed nodes ids to sample from

        Returns:
            Tuple: Return results are the same with Pyg's sampler
        """
        self.lazy_init_quiver()
        
        nodes = input_nodes.to(self.device)
        adjs = []

        batch_size = len(nodes)
        if self.embedding_cache is not None:
            for layer, size in enumerate(self.sizes):
                if layer == 0:
                    out, cnt = self.sample_layer(nodes, size, None)
                    frontier, row_idx, col_idx = self.reindex(nodes, out, cnt)
                    if row_idx.device.type != 'cpu':
                        row_idx, col_idx = row_idx.to('cpu'), col_idx.to('cpu')
                    adj_t = SparseTensor(row=row_idx, col=col_idx, sparse_sizes=(nodes.size(0), frontier.size(0)),
                            is_sorted=True)
                else:
                    tmp_tag = 'layer_' + str(layer)
                    self.embedding_cache[tmp_tag].check_if_fresh(self.batch_count.item())

                    out, cnt = self.sample_layer(nodes, size, self.embedding_cache[tmp_tag].cache_entry_status)
                    # 这里的 out 和 cnt 都没有经过去重, 需要经过去重后再进行 save
                    # out: the total 'global' nID, cnt: the row ptr
                    # print (f"out shape: {out.shape}, cnt shape: {cnt.shape}")
                    frontier, row_idx, col_idx = self.reindex(nodes, out, cnt)

                    # out_test, cnt_test = self.sample_layer(nodes, size, self.invalid_cache)
                    # frontier_test, _, _ = self.reindex(nodes, out_test, cnt_test)
                    # if layer == len(self.sizes) - 1:
                    #     self.reduce += frontier_test.shape[0] - frontier.shape[0]
                    #     self.after += frontier.shape[0]
                    # frontier: global id(去重), row_idx 和 col_idx 对应 local id, 表示邻接关系
                    self.embedding_cache[tmp_tag].evit_and_place_indice(nodes, self.batch_count.item())
                    # reindex still use the out, as we need to put the embedding 'back' to its initial position
                    # row_idx, col_idx = col_idx, row_idx
                    # edge_index = torch.stack([row_idx, col_idx], dim=0)

                    # TODO: more check!
                    start = time.time()
                    if row_idx.device.type != 'cpu':
                        row_idx, col_idx = row_idx.to('cpu'), col_idx.to('cpu')
            
                    adj_t = SparseTensor(row=row_idx, col=col_idx, sparse_sizes=(nodes.size(0), frontier.size(0)),
                            is_sorted=True)  
                    self.save_time += time.time() - start
                size = adj_t.sparse_sizes()[::-1]
                e_id = torch.tensor([])
                adjs.append(Adj(adj_t, e_id, size))
                nodes = frontier
        else:
            for layer, size in enumerate(self.sizes):
                out, cnt = self.sample_layer(nodes, size)
                frontier, row_idx, col_idx = self.reindex(nodes, out, cnt)
                if row_idx.device.type != 'cpu':
                    row_idx, col_idx = row_idx.to('cpu'), col_idx.to('cpu')
                adj_t = SparseTensor(row=row_idx, col=col_idx, sparse_sizes=(nodes.size(0), frontier.size(0)),
                        is_sorted=True)
                size = adj_t.sparse_sizes()[::-1]
                e_id = torch.tensor([])
                adjs.append(Adj(adj_t, e_id, size))
                nodes = frontier
        
        start = time.time()
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1] # reverse
        # print (adjs)
        if frontier.device.type != 'cpu':
            frontier = frontier.to('cpu')
        # TODO: make use of 'transform' in PyG
        # out = (batch_size, nodes, adjs)
        # out = self.transform(*out) if self.transform is not None else out
        
        self.lock.acquire()
        n_id_filename = os.path.join(self.trace_path, self.exp_name, 'sb_' + str(self.sb) + '_ids_' + str(self.batch_count.item()) + '.pth')
        adjs_filename = os.path.join(self.trace_path, self.exp_name, 'sb_' + str(self.sb) + '_adjs_' + str(self.batch_count.item()) + '.pth')
        self.batch_count += 1
        self.lock.release()

        torch.save(frontier, n_id_filename)
        torch.save(adjs, adjs_filename)
        self.save_time += time.time() - start
        # return nodes, batch_size, adjs[::-1]

    def sample_prob(self, train_idx, total_node_count):
        self.lazy_init_quiver()
        last_prob = torch.zeros(total_node_count, device=self.device)
        last_prob[train_idx] = 1
        for size in self.sizes:
            cur_prob = torch.zeros(total_node_count, device=self.device)
            self.quiver.cal_neighbor_prob(0, last_prob, cur_prob, size)
            last_prob = cur_prob
        return last_prob

    def inc_sb(self):
        self.sb += 1
        self.batch_count = torch.zeros(1, dtype=torch.int).share_memory_()
        print("Switch to sb: " + str(self.sb))
    
    def fresh_embedding(self):
        # after each epoch, remove the embeddings
        for layer in range(self.layers):
            tmp_tag = 'layer_' + str(layer + 1)
            self.embedding_cache[tmp_tag].reset()

    def reset_sampler(self):
        self.sb = 0
        self.save_time = 0
        self.after = 0
        self.reduce = 0

        self.batch_count = torch.zeros(1, dtype=torch.int).share_memory_()
        if self.embedding_cache is not None:
            self.fresh_embedding()

    def share_ipc(self):
        """Create ipc handle for multiprocessing

        Returns:
            tuple: ipc handle tuple
        """
        return self.csr_topo, self.sizes, self.mode

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        """Create from ipc handle

        Args:
            ipc_handle (tuple): ipc handle got from calling `share_ipc`

        Returns:
            quiver.pyg.GraphSageSampler: Sampler created from ipc handle
        """
        csr_topo, sizes, mode = ipc_handle
        return cls(csr_topo, sizes, _FakeDevice, mode)

class SampleJob(Generic[T_co]):
    """
    An abstract class representing a :class:`SampleJob`.
    All SampleJobs that represent a map from index to sample tasks should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, :meth:`__getitem__`, :meth:`shuffle`, 
    supporting fetching a sample task for a given index, return the size of the SampleJob and shuffle all tasks in the Job.
    
    """
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def shuffle(self) -> None:
        raise NotImplementedError


def cpu_sampler_worker_loop(rank, quiver_sampler, task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is _StopWork:
            result_queue.put(_StopWork)
            break
        res = quiver_sampler.sample(task)
        result_queue.put(res)

class MixedGraphSageSampler:
    r"""
    Quiver's MixedGraphSageSampler behaves just like `GraphSageSampler` but trying to utilize both GPUs and CPUs to sample graphs.
    It can work in one of [`GPU_CPU_MIXED`, `UVA_CPU_MIXED`, `UVA_ONLY`, `GPU_ONLY`] modes. 
    Args:
        sample_job (SampleJob): A quiver.SampleJob for describing sample tasks
        num_workers (int): Decide parallelism for CPU sampler.
        csr_topo (quiver.CSRTopo): A quiver.CSRTopo for graph topology
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to `sizes[l] = -1`, all neighbors are included
            in layer `l`.
        device (int): Device which sample kernel will be launched
        mode (str): Sample mode, choices are [`GPU_CPU_MIXED`, `UVA_CPU_MIXED`, `UVA_ONLY`, `GPU_ONLY`], default is `UVA_CPU_MIXED`.
    """
    def __init__(self,
                 sample_job: SampleJob,
                 num_workers: int, 
                 csr_topo: quiver_utils.CSRTopo,
                 sizes: List[int],
                 device = 0,
                 mode="UVA_CPU_MIXED"):

        assert mode in ["UVA_CPU_MIXED", "GPU_CPU_MIXED", "UVA_ONLY", "GPU_ONLY"], f"mode should be one of {['UVA_CPU_MIXED', 'GPU_CPU_MIXED', 'UVA_ONLY', 'GPU_ONLY']}"
        
        self.csr_topo = csr_topo
        self.csr_topo.share_memory_()

        self.device = device
        self.sample_job = sample_job
        self.num_workers = num_workers
        self.sizes = sizes
        self.mode = mode

        self.device_quiver = None
        self.cpu_quiver = None

        self.result_queue = None
        self.task_queues = []
        self.device_task_remain = None
        self.cpu_task_remain = None
        self.current_task_id = 0
        self.device_sample_time = 0
        self.cpu_sample_time = 0
        self.device_sample_total = 0
        self.cpu_sample_total = 0

        self.worker_ids = itertools.cycle(range(self.num_workers))
    
        self.inited = False
        self.epoch = 0
    
    def __iter__(self):
        self.sample_job.shuffle()
        if self.epoch <= 1:
            self.device_task_remain = None
            self.cpu_task_remain = None
            self.device_sample_time = 0
            self.cpu_sample_time = 0
            self.device_sample_total = 0
            self.cpu_sample_total = 0
        
        self.current_task_id = 0
        self.epoch += 1
        return self.iter_sampler()

    def decide_task_num(self):
        if self.device_task_remain is None:
            self.device_task_remain = max(20, self.num_workers * 2)
            if self.mode in ["GPU_ONLY", "UVA_ONLY"]:
                self.cpu_task_remain = 0
            else:
                self.cpu_task_remain = self.num_workers
        else:
            self.device_task_remain = max(20, self.num_workers * 2)
            if self.mode in ["GPU_ONLY", "UVA_ONLY"]:
                self.cpu_task_remain = 0
            else:
                self.cpu_task_remain = max(1, int(self.device_sample_time * self.device_task_remain  / self.cpu_sample_time / 2))


        print(f"Device average sample time: {self.device_sample_time}\tCPU average sample time: {self.cpu_sample_time}")
        print(f"Assign {self.device_task_remain} tasks to Device, Assign {self.cpu_task_remain} to CPU")

    def assign_cpu_tasks(self) -> bool:
        for task_id in range(self.current_task_id + self.device_task_remain, self.current_task_id + self.device_task_remain + self.cpu_task_remain):
            if task_id >= len(self.sample_job):
                break
            worker_id = next(self.worker_ids)
            self.task_queues[worker_id].put(self.sample_job[task_id])

    
    def lazy_init(self):
        
        if self.inited:
            return

        self.inited = True
        self.device_quiver = GraphSageSampler(self.csr_topo, self.sizes, device=self.device, mode="GPU" if "GPU" in self.mode else "UVA")
        self.cpu_quiver = GraphSageSampler(self.csr_topo, self.sizes, mode="CPU")
        self.result_queue = mp.Queue()
        for worker_id in range(self.num_workers):
            task_queue = mp.Queue()
            child_process = mp.Process(target=cpu_sampler_worker_loop,
                                       args=(worker_id, self.cpu_quiver, task_queue, self.result_queue))
            child_process.daemon = True
            child_process.start()
            self.task_queues.append(task_queue)

    
    def iter_sampler(self):
        self.lazy_init()
        try:
            while True:
                self.decide_task_num()
                self.assign_cpu_tasks()
                while self.device_task_remain > 0:
                    sample_start = time.time()
                    if self.current_task_id >= len(self.sample_job):
                        break
                    
                    res = self.device_quiver.sample(self.sample_job[self.current_task_id])
                    sample_end = time.time()

                    # Decide average sample time 
                    self.device_sample_time = (self.device_sample_time * self.device_sample_total + sample_end - sample_start) / (self.device_sample_total + 1)
                    self.device_sample_total += 1

                    self.current_task_id += 1
                    self.device_task_remain -= 1
                    yield res

                if self.current_task_id >= len(self.sample_job):
                        break
                while self.cpu_task_remain > 0:
                    sample_start = time.time()
                    res = self.result_queue.get()
                    sample_end = time.time()
                    
                    # Decide average sample time
                    self.cpu_sample_time = (self.cpu_sample_time * self.cpu_sample_total + sample_end - sample_start) / (self.cpu_sample_total + 1)
                    self.cpu_sample_total += 1
                    
                    self.current_task_id += 1
                    self.cpu_task_remain -= 1

                    if self.current_task_id >= len(self.sample_job):
                        break
                    
                    yield res
                
                # Decide to exit
                if self.current_task_id >= len(self.sample_job):
                        break
        except:
            print("something wrong")
            # make sure all child process exit 
            for task_queue in self.task_queues:
                task_queue.put(_StopWork)
            
            for _ in self.task_queues:
                self.result_queue.get()
    
    def share_ipc(self):
        return self.sample_job, self.num_workers, self.csr_topo, self.sizes, self.device, self.mode
    
    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        sample_job, num_workers, csr_topo, sizes, device, mode =ipc_handle
        return cls(sample_job, num_workers, csr_topo, sizes, device, mode)
