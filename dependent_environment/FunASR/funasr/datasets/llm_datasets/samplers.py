import torch
import numpy as np
import logging
import torch.distributed as dist

from funasr.register import tables


@tables.register("batch_sampler_classes", "DynamicBatchLocalShuffleSampler")
class BatchSampler(torch.utils.data.BatchSampler):
    
    def __init__(self, dataset,
                 batch_type: str = "example",
                 batch_size: int = 100,
                 buffer_size: int = 30,
                 drop_last: bool = False,
                 shuffle: bool = True,
                 is_training: bool = True,
                 **kwargs):
        
        self.drop_last = drop_last
        self.pre_idx = -1
        self.dataset = dataset
        self.total_samples = len(dataset)
        self.batch_type = batch_type
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.max_token_length = kwargs.get("max_token_length", 5000)
        self.shuffle_idx = np.arange(self.total_samples)
        self.shuffle = shuffle and is_training
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)
        
    
    def __len__(self):
        return (self.total_samples-1) // self.batch_size + 1
    
    def set_epoch(self, epoch):
        np.random.seed(epoch)
    
    def __iter__(self):
        
        if self.shuffle:
            np.random.shuffle(self.shuffle_idx)
        
        batch = []
        max_token = 0
        num_sample = 0
        
        iter_num = (self.total_samples - 1) // self.buffer_size + 1
        # print("iter_num: ", iter_num)
        for iter in range(self.pre_idx + 1, iter_num):
            datalen_with_index = []
            for i in range(self.buffer_size):
                idx = iter * self.buffer_size + i
                if idx >= self.total_samples:
                    continue
                
                idx_map = self.shuffle_idx[idx]
                # prompt = self.dataset.indexed_dataset[idx_map]["prompt"]
                target_len = self.dataset.get_target_len(idx_map) if self.batch_type == 'length' else 0.0
                source_len = self.dataset.get_source_len(idx_map) / self.length_scale_source
                sample_len_cur = source_len + target_len
                
                
                datalen_with_index.append([idx, sample_len_cur])
            
            datalen_with_index_sort = sorted(datalen_with_index, key=lambda x: x[1])
            for item in datalen_with_index_sort:
                idx, sample_len_cur_raw = item
                if sample_len_cur_raw > self.max_token_length:
                    continue
                
                max_token_cur = max(max_token, sample_len_cur_raw)
                max_token_padding = 1 + num_sample
                if self.batch_type != 'example':
                    max_token_padding *= max_token_cur
                if max_token_padding <= self.batch_size:
                    batch.append(idx)
                    max_token = max_token_cur
                    num_sample += 1
                else:
                    yield batch
                    batch = [idx]
                    max_token = sample_len_cur_raw
                    num_sample = 1


@tables.register("batch_sampler_classes", "BatchSampler")
@tables.register("batch_sampler_classes", "RankFullLocalShuffleBatchSampler")
class RankFullLocalShuffleBatchSampler(torch.utils.data.BatchSampler):
    
    def __init__(self, dataset,
                 batch_type: str = "example",
                 batch_size: int = 100,
                 buffer_size: int = 30,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 is_training: bool = True,
                 **kwargs):
        
        self.drop_last = drop_last
        self.pre_idx = -1
        self.dataset = dataset
        self.total_samples = len(dataset)
        self.batch_type = batch_type
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.shuffle_idx = np.arange(self.total_samples)
        self.shuffle = shuffle and is_training
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)
        
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
        self.rank = rank
        self.world_size = world_size
        
    def __len__(self):
        return (self.total_samples - 1) // (self.batch_size * self.world_size) + 1
    
    def set_epoch(self, epoch):
        np.random.seed(epoch)
    
    def __iter__(self):
    
        batch_size_total = self.batch_size * self.world_size
        
        if self.shuffle:
            np.random.shuffle(self.shuffle_idx)
        
        batch = []
        max_token = 0
        num_sample = 0
        
        iter_num = (self.total_samples - 1) // self.buffer_size + 1
        # print("iter_num: ", iter_num)
        for iter in range(self.pre_idx + 1, iter_num):
            # if iter == iter_num -1 and self.drop_last:
            #     continue
            datalen_with_index = []
            for i in range(self.buffer_size):
                idx = iter * self.buffer_size + i
                if idx >= self.total_samples:
                    continue
                
                idx_map = self.shuffle_idx[idx]
                # prompt = self.dataset.indexed_dataset[idx_map]["prompt"]
                
                source_len = self.dataset.get_source_len(idx_map) / self.length_scale_source
                target_len = self.dataset.get_target_len(idx_map) if self.batch_type == 'length' else 0.0
                sample_len_cur = source_len + target_len
                
                datalen_with_index.append([idx, sample_len_cur])
            
            datalen_with_index_sort = sorted(datalen_with_index, key=lambda x: x[1])
            for item in datalen_with_index_sort:
                idx, sample_len_cur_raw = item
                if sample_len_cur_raw > self.max_token_length:
                    continue

                max_token_cur = max(max_token, sample_len_cur_raw)
                max_token_padding = 1 + num_sample
                # if self.batch_type != 'example':
                #     max_token_padding *= max_token_cur
                if max_token_padding <= batch_size_total:
                    batch.append(idx)
                    max_token = max_token_cur
                    num_sample += 1
                else:
                    batch_rank = batch[self.rank*self.batch_size: (self.rank+1)*self.batch_size]
                    yield batch_rank
                    batch = [idx]
                    max_token = sample_len_cur_raw
                    num_sample = 1


@tables.register("batch_sampler_classes", "RankFullLocalShuffleDynamicBatchSampler")
class RankFullLocalShuffleDynamicBatchSampler(torch.utils.data.BatchSampler):
    
    def __init__(self, dataset,
                 batch_type: str = "example",
                 batch_size: int = 100,
                 buffer_size: int = 30,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 is_training: bool = True,
                 **kwargs):
        
        self.drop_last = drop_last
        self.pre_idx = -1
        self.dataset = dataset
        self.total_samples = len(dataset)
        self.batch_type = batch_type
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.shuffle_idx = np.arange(self.total_samples)
        self.shuffle = shuffle and is_training
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)
        
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
        self.rank = rank
        self.world_size = world_size
    
    def __len__(self):
        return (self.total_samples - 1) // (self.batch_size * self.world_size) + 1
    
    def set_epoch(self, epoch):
        np.random.seed(epoch)
    
    def __iter__(self):
        
        batch_size_total = self.batch_size * self.world_size
        if self.shuffle:
            np.random.shuffle(self.shuffle_idx)
        
        batch_list_all_rank = []
        batch_list_cur = []
        max_token = 0
        num_sample = 0
        
        iter_num = (self.total_samples - 1) // self.buffer_size + 1
        # print("iter_num: ", iter_num)
        for iter in range(self.pre_idx + 1, iter_num):
            # if iter == iter_num - 1 and self.drop_last:
            #     continue
            datalen_with_index = []
            for i in range(self.buffer_size):
                idx = iter * self.buffer_size + i
                if idx >= self.total_samples:
                    continue
                
                idx_map = self.shuffle_idx[idx]
                # prompt = self.dataset.indexed_dataset[idx_map]["prompt"]
                
                source_len = self.dataset.get_source_len(idx_map) / self.length_scale_source
                target_len = self.dataset.get_target_len(idx_map) if self.batch_type == 'length' else 0.0
                sample_len_cur = source_len + target_len
                
                datalen_with_index.append([idx, sample_len_cur])
            
            datalen_with_index_sort = sorted(datalen_with_index, key=lambda x: x[1])
            for ii, item in enumerate(datalen_with_index_sort):
                is_last_batch = iter == iter_num - 1 and ii == len(datalen_with_index_sort)
                idx, sample_len_cur_raw = item
                if sample_len_cur_raw > self.max_token_length:
                    continue
                
                max_token_cur = max(max_token, sample_len_cur_raw)
                max_token_padding = 1 + num_sample
                
                if self.batch_type != 'example':
                    max_token_padding *= max_token_cur
                if len(batch_list_all_rank) < self.world_size:
                    
                    if max_token_padding <= self.batch_size:
                        batch_list_cur.append(idx)
                        max_token = max_token_cur
                        num_sample += 1
                    else:
                        batch_list_all_rank.append(batch_list_cur)
                        batch_list_cur = []
                else:
                    batch_rank = batch_list_all_rank[self.rank]
                    yield batch_rank
                    batch_list_all_rank = [idx]
                    max_token = sample_len_cur_raw
                    num_sample = 1
