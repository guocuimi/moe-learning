import torch
import torch.distributed as dist
import time

# torchrun --nproc_per_node 4 ./a2a_split_2_test.py
if __name__ == "__main__":
    
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    size = dist.get_world_size()
    
    device = torch.device("cuda", rank)
    torch.manual_seed(rank)
    
    input_splits = torch.randint(1, 6, (size,), dtype=torch.int32, device=device).tolist()
    input = torch.ones(sum(input_splits), dtype=torch.int32, device=device) * rank 
    
    time.sleep(rank * 0.1)
    print(f"Rank {rank}, input_splits: {input_splits}")
    
    # communicate input splits to all ranks
    input_splits_pt = torch.tensor(input_splits, dtype=torch.int32, device=device)
    output_splits_pt = torch.empty(size, dtype=torch.int32, device=device)
    dist.all_to_all_single(output_splits_pt, input_splits_pt)
    output_splits = output_splits_pt.cpu().tolist()
    
    time.sleep(rank * 0.1)
    print(f"Rank {rank}, output_splits: {output_splits}")
    
    time.sleep(1)
    if rank == 0:
        print("--------------------------------")
    
    time.sleep(rank * 0.1)
    print(f"Rank {rank}, input: {input.tolist()}")
    output = torch.zeros(sum(output_splits), dtype=torch.int32, device=device)
    dist.all_to_all_single(output=output, input=input, output_split_sizes=output_splits, input_split_sizes=input_splits)
    
    time.sleep(1)
    
    time.sleep(rank * 0.1)
    print(f"Rank {rank}, output: {output.tolist()}")

    time.sleep(1)
    if rank == 0:
        print("--------------------------------")

    input2 = torch.empty_like(input)
    dist.all_to_all_single(output=input2, input=output, output_split_sizes=input_splits, input_split_sizes=output_splits)

    time.sleep(rank * 0.2)
    print(f"Rank {rank}, input2: {input2.tolist()}")

    time.sleep(1)

    if torch.equal(input, input2):
        print(f"Rank {rank}, Success")
    else:
        print(f"Rank {rank}, Failed")
    
    dist.destroy_process_group()