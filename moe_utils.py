import torch

def permute(tokens, indices, topk: int = 1):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens, topk].
        topk (int, optional): The topk value. Defaults to 1.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    if topk > 1:
        assert indices.size(1) == topk
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, sorted_indices, probs: torch.Tensor = None, topk: int = 1):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
        sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
        probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will be merged with their respective probabilities.
        topk (int, optional): The number of top tokens to consider for merging with probabilities. Defaults to 1.
    """
    if topk > 1:
        assert probs is not None
        assert (
            probs.size(0) == permuted_tokens.size(0) // topk
        ), f"{probs.size()} {permuted_tokens.size()}"
    if probs is not None:
        assert probs.size(0) == permuted_tokens.size(0) // topk
        assert probs.size(1) == topk, f"probs size {probs.size()} merge_factor {topk}"

    unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)

    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens