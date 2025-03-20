import torch

n_experts = 8
n_local_experts = 4

# [n_experts]: [0, 1, 2, 3, 4, 5, 6, 7]
input_chunk_idxs = torch.arange(n_experts)

print(f"input_chunk_idxs: {input_chunk_idxs}")

# [n_local_experts, ep_size]
#  => [[0, 2, 4, 6], [1, 3, 5, 7]]
#  => [0, 2, 4, 6, 1, 3, 5, 7]
sort_by_local_experts = input_chunk_idxs.reshape(-1, n_local_experts).T.ravel()
print(f"sort_by_local_experts: {sort_by_local_experts}")

# [n_local_experts, ep_size]
#  => [[0, 1, 2, 3], [4, 5, 6, 7]]
#  => [[0, 4], [1, 5], [2, 6], [3, 7]]
#  => [0, 4, 1, 5, 2, 6, 3, 7]
restore_output_by_local_experts = input_chunk_idxs.reshape(
    n_local_experts, -1
).T.ravel()

print(f"restore_output_by_local_experts: {restore_output_by_local_experts}")
