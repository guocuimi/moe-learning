import time
import torch
import torch.distributed as dist


def permute(
    tokens: torch.Tensor,  # [n_tokens, dim]
    routing_map: torch.Tensor,  # [n_tokens, n_experts]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_tokens, n_experts = routing_map.shape

    routing_map = routing_map.bool().T.contiguous()
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
    return permuted_tokens, sorted_indices, routing_map


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
    return output.to(dtype=permuted_tokens.dtype)


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
        # [n_experts, n_tokens]
        self.routing_map = None
        # [n_permuted_tokens, n_tokens]
        self.permuted_probs = None

        # [ep_size]
        self.input_splits = None
        self.output_splits = None

    def preprocess(self, routing_map: torch.Tensor):
        # [n_tokens, n_experts] => [n_experts]
        local_tokens_per_expert = routing_map.sum(dim=0)
        # calculate input_splits, output_splits for all2all communication
        # [n_experts] => [ep_size, n_local_experts] => [ep_size]
        # num of tokens for each rank
        self.input_splits = local_tokens_per_expert.reshape(
            self.ep_size, self.n_local_experts
        ).sum(dim=1)

        time.sleep(0.2 * self.ep_rank)
        print(f"Rank {rank}, input_splits: {self.input_splits}")
        # print(f"Rank {rank}, local_tokens_per_expert: {local_tokens_per_expert}")

        # gather the global distribution of tokens accross ranks/devices
        # [n_experts] => [ep_size, n_experts]
        tokens_per_expert = gather_along_first_dim(
            self.ep_group, local_tokens_per_expert
        ).reshape(self.ep_size, self.n_experts)

        # time.sleep(1 * self.ep_rank)
        # print(f"Rank {rank}, tokens_per_expert: {tokens_per_expert}")

        # tokens per local expert
        # [ep_size, n_experts] => [ep_size, n_local_experts]
        tokens_per_local_expert = tokens_per_expert[
            :,
            self.ep_rank * self.n_local_experts : (self.ep_rank + 1)
            * self.n_local_experts,
        ]

        # print(f"Rank {rank}, tokens_per_local_expert: {tokens_per_local_expert}")

        # [ep_size, n_local_experts] => [ep_size]
        tokens_per_rank = tokens_per_local_expert.sum(dim=1)
        # print(f"Rank {rank}, tokens_per_rank: {tokens_per_rank}")

        # [ep_size]
        self.output_splits = tokens_per_rank

        time.sleep(1 * self.ep_rank)
        print(f"Rank {rank}, output_splits: {self.output_splits}")

        # [ep_size, n_local_experts] => [n_local_experts]
        tokens_per_local_expert = tokens_per_local_expert.sum(dim=0)
        return tokens_per_local_expert

    def token_permutation(
        self,
        tokens: torch.Tensor,  # [n_tokens, dim]
        probs: torch.Tensor,  # [n_tokens, n_experts]
        routing_map: torch.Tensor,  # [n_tokens, n_experts]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.restore_shape = tokens.shape
        tokens_per_local_expert = self.preprocess(routing_map)

        # Permute the tokens based on the routing map.
        # Tokens with the same designated expert will be grouped together.
        permuted_tokens, self.sorted_indices, self.routing_map = permute(
            tokens, routing_map
        )
        # [n_experts, n_tokens] -> [n_permuted_tokens]
        self.permuted_probs = probs.T.contiguous().masked_select(mask=self.routing_map)
        return permuted_tokens, tokens_per_local_expert

    def token_unpermutation(
        self,
        permuted_tokens: torch.Tensor,  # [n_tokens * n_topk, dim]
    ) -> torch.Tensor:
        # [n_permuted_tokens, dim]
        permuted_tokens *= self.permuted_probs.unsqueeze(-1)
        # Restore the original order of tokens after permutation
        output = unpermute(permuted_tokens, self.sorted_indices, self.restore_shape)
        return output


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
    n_topk = 3
    dim = 8
    tokens = torch.randn((n_tokens, dim))
    logits = torch.randn((n_tokens, n_experts))
    # [n_tokens, n_topk]
    expert_weights, expert_indices = torch.topk(logits, n_topk, dim=-1)

    # print("expert_weights:", expert_weights)
    # print("expert_indices:", expert_indices)

    expert_weights = torch.softmax(expert_weights, dim=-1)
    # expert_indices to dense mask: [n_tokens, n_experts]
    routing_map = (
        torch.zeros_like(logits)
        .int()
        .scatter(dim=1, index=expert_indices, value=1)
        .bool()
    )
    probs = torch.zeros_like(logits).scatter(
        dim=1, index=expert_indices, src=expert_weights
    )

    # print("routing_map:", routing_map)
    # print("probs:", probs)

    dispatcher = TokenDispatcher(
        ep_rank=rank,
        ep_size=world_size,
        n_experts=n_experts,
        n_local_experts=n_experts // world_size,
        ep_group=ep_group,
    )

    # print("tokens:", tokens)
    permuted_input, sorted_indices = dispatcher.token_permutation(
        tokens, probs, routing_map
    )

    # print("permuted_input:", permuted_input)
    # print("sorted_indices:", sorted_indices)

    # unpermute
    output = dispatcher.token_unpermutation(permuted_input)
    # print("output:", output)
    time.sleep(0.5 * rank)
    if torch.allclose(tokens, output):
        print(f"Rank {rank}, Success")
    else:
        print(f"Rank {rank}, Fail")

    # distroy the process group
    dist.destroy_process_group()
