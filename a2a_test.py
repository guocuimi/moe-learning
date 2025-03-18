import torch
import torch.distributed as dist
import time

# torchrun --nproc_per_node 4 ./a2a_test.py
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    size = dist.get_world_size()
    
    device = torch.device("cuda", rank)
    input = torch.ones(size, dtype=torch.int32, device=device) * rank 
    time.sleep(rank * 0.1)
    print(f"Rank {rank}, input: {input.tolist()}")
    
    output = torch.empty(size, dtype=torch.int32, device=device)
    dist.all_to_all_single(output, input)
    
    time.sleep(rank * 0.1)
    print(f"Rank {rank}, output: {output.tolist()}")
    
    dist.destroy_process_group()