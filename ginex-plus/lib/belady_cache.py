import torch
import numpy as np
import tqdm

class beladyCache: 
    def __init__(self, effective_sb_size, n_id_list, num_nodes, cache_entries):
        
        self.n_id_list = n_id_list
        self.effective_sb_size = effective_sb_size
        # The current implementation use int16 for 'iters' which limits the number of 
        # iterations to perform cache state simulation.
        if self.effective_sb_size > torch.iinfo(torch.int16).max:
            raise ValueError
        self.num_nodes = num_nodes
        self.num_entries = cache_entries

        # The address table of the cache has num_nodes entries each of which is a single
        # int32 value. This can support the cache with up to 2147483647 entries.
        
        if self.num_entries > torch.iinfo(torch.int32).max:
            raise ValueError


    # Two passes over ids files to construct data structures for cache state simulation and
    # figure out the initial cache indices.
    def pass_1_and_2(self):
        
        frq = torch.zeros(self.num_nodes, dtype=torch.int16, device='cuda')
        filled=False
        count=0
        initial_cache_indices = torch.empty((0,), dtype=torch.int64, device='cuda')
        for n_id in self.n_id_list:
            n_id = n_id.cuda()
            if not filled:
                to_cache = n_id[frq[n_id] == 0]
                count += to_cache.numel()
                if count >= self.num_entries:
                    to_cache = to_cache[:self.num_entries-(count-to_cache.numel())]
                    initial_cache_indices = torch.cat([initial_cache_indices, to_cache])
                    filled=True
                else:
                    initial_cache_indices = torch.cat([initial_cache_indices, to_cache])

            frq[n_id] += 1

        msb = (torch.tensor([1], dtype=torch.int16) << 15).cuda()

        cumsum = frq.cumsum(dim=0)
        iterptr = torch.cat([torch.tensor([0,0], device='cuda'), cumsum[:-1]]); del(cumsum)
        frq_sum = frq.sum(); del(frq)

        iters = torch.zeros(frq_sum+1, dtype=torch.int16, device='cuda')
        iters[-1] = self.effective_sb_size

        for i, n_id in enumerate(self.n_id_list):
            n_id_cuda = n_id.cuda()
            tmp = iterptr[n_id_cuda+1]
            iters[tmp] = i; del(tmp)
            iterptr[n_id_cuda+1] += 1; del(n_id_cuda)
        iters[iterptr[1:]] |= msb
        iterptr = iterptr[:-1]
        iterptr[0] = 0


        return iterptr, iters, initial_cache_indices


    # The last pass over ids files to simulate the cache state.
    def pass_3(self, iterptr, iters, initial_cache_indices):

        # Two auxiliary data structures for efficient cache state simulation
        #
        # cache_table: a table recording each node's state which is updated every
        #   iteration of the simulation. It has entries as many as the total number 
        #   of the nodes in the graph. Specifically, the last three bits of each 
        #   entires are used.
        #   bit 0: set if the feature vector of the corresponindg node is in the  
        #   cache at the current iteration
        #   bit 1: set if the feature vector of the corresponding node is accessed  
        #   at the current interation
        #   bit 2: set if the node is selected to be kept in the cache for the next 
        #   iteration
        #   bit 3~7: don't care
        #
        # map_table: mapping table that directly maps relative indices of the feature 
        #   vectors in the batch inputs, which is the output of gather, to their 
        #   absolute indices

        cache_table = torch.zeros(self.num_nodes, dtype=torch.int8, device='cuda')
        cache_table[initial_cache_indices] += 1; del(initial_cache_indices)
        
        msb = (torch.tensor([1], dtype=torch.int16) << 15).cuda()

        save_p = None
        threshold = 0


        cache_rate = 0.0
        for iter in range(len(self.n_id_list)):
            n_id = self.n_id_list[iter]
            n_id_cuda = n_id.cuda()
            del(n_id)

            # Update iterptr
            iterptr[n_id_cuda] += 1
            last_access = n_id_cuda[(iters[iterptr[n_id_cuda]] < 0)]
            iterptr[last_access] = iters.numel()-1; del(last_access)

            # Get candidates
            # candidates = union(current cache indices, incoming indices)
            cache_table[n_id_cuda] += 2
            candidates = (cache_table > 0).nonzero().squeeze(); del(n_id_cuda)

            # Get next access iterations of candidates
            next_access_iters = iters[iterptr[candidates]]
            next_access_iters.bitwise_and_(~msb)
            
            # Find num_entries elements in candidates with the smallest next access
            # iteration by incrementally tracking threshold
            count = (next_access_iters <= threshold).sum()
            prev_status = (count >= self.num_entries)

            if prev_status:
                # Current threshold is high
                threshold -= 1
            else:
                # Current threshold is low
                threshold += 1
            while (True):
                if threshold > self.effective_sb_size:
                    num_remains = 0
                    break

                count = (next_access_iters <= threshold).sum()
                curr_status = (count >= self.num_entries)
                if (prev_status ^ curr_status):
                    if curr_status:
                        num_remains = self.num_entries - (next_access_iters <= (threshold-1)).sum()
                        threshold -= 1
                    else:
                        num_remains = self.num_entries - count
                    break
                elif (curr_status): threshold -= 1
                else: threshold += 1
            
            cache_table[candidates[next_access_iters <= threshold]] |= 4
            cache_table[candidates[next_access_iters == (threshold+1)][:num_remains]] |= 4
            del(candidates)
            del(next_access_iters)
            
            # in_indices: indices to newly insert into cache
            # in_positions: relative positions of nodes in in_indices within batch input
            # out_indices: indices to evict from cache
            
            # in_indices = (cache_table == 2+4).nonzero().squeeze()
            # remain_indices = (cache_table == 7).nonzero().squeeze()
            # out_indices = ((cache_table == 1) | (cache_table == 3)).nonzero().squeeze()
            cur_indices = (cache_table >= 4).nonzero().squeeze()
            
            if iter < len(self.n_id_list) - 1:
                if cache_rate == 0.0:
                    cache_rate = float(len(set(self.n_id_list[iter + 1].tolist()) & set(cur_indices.tolist()))) / len(self.n_id_list[iter + 1].tolist())
                else:
                    cache_rate += float(len(set(self.n_id_list[iter + 1].tolist()) & set(cur_indices.tolist()))) / len(self.n_id_list[iter + 1].tolist())
            
            cache_table >>= 2 

            #####################################################################

        # del(cache_table)
        # del(iterptr)
        # del(iters)
        print ("{:.4f}% average cache rate.".format(cache_rate / (self.effective_sb_size - 1) * 100))
        return
    
    def simulate(self):
        iterptr, iter, initial_cache_indices = self.pass_1_and_2()
        self.pass_3(iterptr, iter, initial_cache_indices)