import torch


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


class TokenDispatcher:
    def __init__(self):
        # store the original shape for unpermutation
        self.restore_shape = None
        # original token indices for permutated tokens
        self.sorted_indices = None
        # [n_experts, n_tokens]
        self.routing_map = None
        # [n_permuted_tokens, n_tokens]
        self.permuted_probs = None

    def token_permutation(
        self,
        tokens: torch.Tensor,  # [n_tokens, dim]
        probs: torch.Tensor,  # [n_tokens, n_experts]
        routing_map: torch.Tensor,  # [n_tokens, n_experts]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.restore_shape = tokens.shape

        # [n_tokens, n_experts] => [n_experts]
        local_tokens_per_expert = routing_map.sum(dim=0)

        # Permute the tokens based on the routing map.
        # Tokens with the same designated expert will be grouped together.
        permuted_tokens, self.sorted_indices, self.routing_map = permute(
            tokens, routing_map
        )
        # [n_experts, n_tokens] -> [n_permuted_tokens]
        self.permuted_probs = probs.T.contiguous().masked_select(mask=self.routing_map)
        return permuted_tokens, local_tokens_per_expert

    def token_unpermutation(
        self,
        permuted_tokens: torch.Tensor,  # [n_tokens * n_topk, dim]
    ) -> torch.Tensor:
        # [n_permuted_tokens, dim]
        permuted_tokens *= self.permuted_probs.unsqueeze(-1)
        # Restore the original order of tokens after permutation
        output = unpermute(permuted_tokens, self.sorted_indices, self.restore_shape)
        return output


if __name__ == "__main__":
    ep_group = None

    rank = 0
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

    dispatcher = TokenDispatcher()

    # print("tokens:", tokens)
    permuted_input, sorted_indices = dispatcher.token_permutation(
        tokens, probs, routing_map
    )

    # print("permuted_input:", permuted_input)
    # print("sorted_indices:", sorted_indices)

    # unpermute
    output = dispatcher.token_unpermutation(permuted_input)
    # print("output:", output)
    if torch.allclose(tokens, output):
        print("Success")
    else:
        print("Fail")
