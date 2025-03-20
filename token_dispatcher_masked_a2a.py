import time
import torch
import torch.distributed as dist


def permute(
    tokens: torch.Tensor,  # [n_tokens, dim]
    routing_map: torch.Tensor,  # [n_experts, n_tokens]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_experts, n_tokens = routing_map.shape

    # [n_experts, n_tokens]
    token_indices = (
        torch.arange(n_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(n_experts, -1)
    )
    # original token indices, sorted by expert idx
    sorted_indices = token_indices.masked_select(mask=routing_map)
    # [n_permuted_tokens, dim]
    permuted_tokens = tokens.index_select(dim=0, index=sorted_indices)
    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,  # [n_permuted_tokens, dim]
    sorted_indices: torch.Tensor,  # [n_permuted_tokens]
    restore_shape: torch.Size,  # [n_tokens, dim]
) -> torch.Tensor:
    _, hidden = restore_shape
    output = torch.zeros(
        restore_shape,
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    output.scatter_add_(
        dim=0,
        index=sorted_indices.unsqueeze(1).expand(-1, hidden),
        src=permuted_tokens,
    )
    return output


# sort the chunks based on the sorted indices
def sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: list,
    sorted_idxs: list,
):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes, dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output


# returns [world_size, n_experts]
def gather_along_first_dim(
    ep_group,
    input: torch.Tensor,  # [n_experts]
) -> torch.Tensor:
    """Gather tensors and concatenate along the first dimension"""
    world_size = dist.get_world_size(ep_group)
    if world_size == 1:
        return input
    dim_size = list(input.size())
    dim_size[0] *= world_size
    output = torch.empty(dim_size, dtype=input.dtype, device=input.device)
    dist.all_gather_into_tensor(output, input, group=ep_group)
    return output


def all_to_all(
    ep_group,
    input: torch.Tensor,  # [n_tokens, dim]
    output_split_sizes: list,  # [ep_size]
    input_split_sizes: list,  # [ep_size]
):
    world_size = torch.distributed.get_world_size(ep_group)
    if world_size == 1:
        return input
    # ensure input is contiguous
    input = input.contiguous()
    # [sum(output_split_sizes), dim]
    output = input.new_empty(
        size=[sum(output_split_sizes)] + list(input.size()[1:]),
    )
    torch.distributed.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=ep_group,
    )
    return output


class TokenDispatcher:
    def __init__(
        self,
        ep_rank: int,
        ep_size: int,
        n_experts: int,
        n_local_experts: int,
        ep_group,
    ):
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.n_experts = n_experts
        self.n_local_experts = n_local_experts
        assert ep_size * n_local_experts == n_experts

        self.ep_group = ep_group

        # store the original shape for unpermutation
        self.restore_shape = None
        # original token indices for permutated tokens
        self.sorted_indices = None
        # [n_permuted_tokens, n_tokens]
        self.permuted_probs = None

        # [ep_size]
        self.input_splits = None
        self.output_splits = None

        # [ep_size, n_local_experts]
        self.tokens_per_rank_per_local_expert = None

        # [n_experts]: [0, 1, 2, 3, 4, 5, 6, 7]
        input_chunk_idxs = torch.arange(n_experts)

        # [ep_size, n_local_experts]
        #     [0, 1, 2, 3, 4, 5, 6, 7]
        #  => [[0, 1], [2, 3], [4, 5], [6, 7]]
        #  => [[0, 2, 4, 6], [1, 3, 5, 7]]
        #  => [0, 2, 4, 6, 1, 3, 5, 7]
        self.sort_by_local_experts = (
            input_chunk_idxs.reshape(-1, n_local_experts).T.ravel().tolist()
        )

        # [n_local_experts, ep_size]
        #     [0, 1, 2, 3, 4, 5, 6, 7]
        #  => [[0, 1, 2, 3], [4, 5, 6, 7]]
        #  => [[0, 4], [1, 5], [2, 6], [3, 7]]
        #  => [0, 4, 1, 5, 2, 6, 3, 7]
        self.restore_output_by_local_experts = (
            input_chunk_idxs.reshape(n_local_experts, -1).T.ravel().tolist()
        )

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        # [n_tokens, n_experts] => [n_experts]
        local_tokens_per_expert = routing_map.sum(dim=0)
        # calculate input_splits, output_splits for all2all communication
        # [n_experts] => [ep_size, n_local_experts] => [ep_size]
        # num of tokens for each rank
        self.input_splits = (
            local_tokens_per_expert.reshape(self.ep_size, self.n_local_experts)
            .sum(dim=1)
            .tolist()
        )

        # gather the global distribution of tokens accross ranks/devices
        # [n_experts] => [ep_size, n_experts]
        tokens_per_expert = gather_along_first_dim(
            self.ep_group, local_tokens_per_expert
        ).reshape(self.ep_size, self.n_experts)

        # tokens per local expert
        # [ep_size, n_experts] => [ep_size, n_local_experts]
        tokens_per_rank_per_local_expert = tokens_per_expert[
            :,
            self.ep_rank * self.n_local_experts : (self.ep_rank + 1)
            * self.n_local_experts,
        ]

        # [ep_size, n_local_experts]
        self.tokens_per_rank_per_local_expert = tokens_per_rank_per_local_expert

        # print(f"Rank {rank}, tokens_per_local_expert: {tokens_per_local_expert}")

        # [ep_size, n_local_experts] => [ep_size]
        tokens_per_rank = tokens_per_rank_per_local_expert.sum(dim=1)
        # print(f"Rank {rank}, tokens_per_rank: {tokens_per_rank}")

        # [ep_size]
        self.output_splits = tokens_per_rank.tolist()

        # [ep_size, n_local_experts] => [n_local_experts]
        tokens_per_local_expert = tokens_per_rank_per_local_expert.sum(dim=0)

        return tokens_per_local_expert

    def token_permutation(
        self,
        tokens: torch.Tensor,  # [n_tokens, dim]
        probs: torch.Tensor,  # [n_tokens, n_experts]
        routing_map: torch.Tensor,  # [n_tokens, n_experts]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.restore_shape = tokens.shape
        tokens_per_local_expert = self.preprocess(routing_map)

        # [n_tokens, n_experts] => [n_experts, n_tokens]
        routing_map = routing_map.T
        # Permute local tokens based on the routing map.
        # Tokens with the same designated expert will be grouped together.
        local_permuted_tokens, self.sorted_indices = permute(tokens, routing_map)
        # [n_experts, n_tokens] -> [n_permuted_tokens]
        self.permuted_probs = probs.T.contiguous().masked_select(mask=routing_map)

        # all2all communication
        global_permuted_tokens = all_to_all(
            self.ep_group,
            input=local_permuted_tokens,
            output_split_sizes=self.output_splits,
            input_split_sizes=self.input_splits,
        )

        if self.n_local_experts > 1:
            # sort tokens by (n_local_experts, ep_size)
            global_permuted_tokens = sort_chunks_by_idxs(
                global_permuted_tokens,
                self.tokens_per_rank_per_local_expert.ravel().tolist(),
                self.sort_by_local_experts,
            )
        return global_permuted_tokens, tokens_per_local_expert

    def token_unpermutation(
        self,
        global_permuted_tokens: torch.Tensor,  # [n_permuted_tokens, dim]
    ) -> torch.Tensor:
        if self.n_local_experts > 1:
            # sort tokens by (ep_size, n_local_experts)
            global_permuted_tokens = sort_chunks_by_idxs(
                global_permuted_tokens,
                self.tokens_per_rank_per_local_expert.T.ravel().tolist(),
                self.restore_output_by_local_experts,
            )

        # all2all communication
        local_permuted_tokens = all_to_all(
            self.ep_group,
            input=global_permuted_tokens,
            output_split_sizes=self.input_splits,
            input_split_sizes=self.output_splits,
        )

        # [n_permuted_tokens, dim]
        local_permuted_tokens *= self.permuted_probs.unsqueeze(-1)
        # Restore the original order of tokens after permutation
        return unpermute(local_permuted_tokens, self.sorted_indices, self.restore_shape)


# torchrun --nproc_per_node 4 ./token_dispatcher_masked_a2a.py
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # use default group
    ep_group = None

    device = torch.device("cuda", rank)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
    torch.manual_seed(rank)

    n_tokens = 3
    n_experts = 8
    assert n_experts % world_size == 0

    n_topk = 3
    dim = 8
    tokens = torch.randn((n_tokens, dim))
    logits = torch.randn((n_tokens, n_experts))
    # [n_tokens, n_topk]
    expert_weights, expert_indices = torch.topk(logits, n_topk, dim=-1)
    expert_weights = torch.softmax(expert_weights, dim=-1)
    # convert expert_indices to dense mask: [n_tokens, n_experts]
    routing_map = (
        torch.zeros_like(logits, dtype=torch.int)
        .scatter(dim=1, index=expert_indices, value=1)
        .bool()
    )
    probs = torch.zeros_like(logits).scatter(
        dim=1, index=expert_indices, src=expert_weights
    )

    dispatcher = TokenDispatcher(
        ep_rank=rank,
        ep_size=world_size,
        n_experts=n_experts,
        n_local_experts=n_experts // world_size,
        ep_group=ep_group,
    )

    permuted_input, sorted_indices = dispatcher.token_permutation(
        tokens, probs, routing_map
    )

    # unpermute
    output = dispatcher.token_unpermutation(permuted_input)

    time.sleep(0.5 * rank)
    if torch.allclose(tokens, output):
        print(f"Rank {rank}, Success")
    else:
        print(f"Rank {rank}, Fail")

    # distroy the process group
    dist.destroy_process_group()
