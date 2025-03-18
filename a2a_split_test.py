import torch
import torch.distributed as dist
import time

# torchrun --nproc_per_node 4 ./a2a_split_test.py
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    size = dist.get_world_size()
    
    device = torch.device("cuda", rank)
    # input_splits = [1, 2, 3, 4]
    input_splits = [i + 1 for i in range(size)]
    # output_splits = [1, 1, 1, 1] * (rank + 1)
    output_splits = [rank + 1] * size
    
    input = torch.ones(sum(input_splits), dtype=torch.int32, device=device) * rank 
    time.sleep(rank * 0.1)
    print(f"Rank {rank}, splits: {input_splits}, input: {input.tolist()}")
    
    output = torch.empty(sum(output_splits), dtype=torch.int32, device=device)
    dist.all_to_all_single(output, input, output_splits, input_splits)
    
    time.sleep(rank * 0.1)
    print(f"Rank {rank}, output_splits: {output_splits}, output: {output.tolist()}")
    
    dist.destroy_process_group()