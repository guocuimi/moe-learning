from typing import Optional
import torch


def permute(
    tokens: torch.Tensor,  # [n_tokens, hidden]
    routing_map: torch.Tensor,  # [n_tokens, n_experts]
):
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    # [num_experts, num_tokens]
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(num_experts, -1)
    )
    # sorted by expert
    sorted_indices = token_indices.masked_select(mask=routing_map)

    # use the mapping to permute the tokens
    # [num_experts * num_tokens, hidden]
    permuted_input = tokens.index_select(dim=0, index=sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,  # [n_tokens * n_topk, dim]
    sorted_indices: torch.Tensor,  # [n_tokens * n_topk]
    restore_shape: torch.Size,  # [n_tokens, dim]
    probs: torch.Tensor = None,  # [n_tokens, n_experts]
    routing_map: torch.Tensor = None,  # [n_tokens, n_experts]
):
    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."

        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    # Scatter add the permuted_input back to the original positions
    # self[index[i][j]][j] += src[i][j]
    output_tokens.scatter_add_(
        dim=0, index=sorted_indices.unsqueeze(1).expand(-1, hidden), src=permuted_tokens
    )
    return output_tokens.to(dtype=input_dtype)


def sort_chunks_by_idxs(
    input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor
):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    return output


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    # torch.set_default_device("cuda")
    torch.manual_seed(0)

    n_tokens = 3
    n_experts = 8
    n_activated_experts = 2
    dim = 4
    x = torch.randn((n_tokens, dim))
    logits = torch.randn((n_tokens, n_experts))
    # [n_tokens, n_topk]
    expert_weights, expert_indices = torch.topk(logits, n_activated_experts, dim=-1)
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

    print("x:", x)
    permuted_input, sorted_indices = permute(x, routing_map)

    print("permuted_input:", permuted_input)
    print("sorted_indices:", sorted_indices)

    # unpermute
    output = unpermute(
        permuted_input, sorted_indices, x.shape, probs=probs, routing_map=routing_map
    )
    print("output:", output)
