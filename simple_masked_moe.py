from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

use_token_dispatcher = False


@dataclass
class ModelArgs:
    dim: int = 16
    intermediate_size: int = 64
    num_experts: int = 8
    router_topk: int = 2
    num_shared_experts: int = 1


class TokenDispatcher:
    def __init__(self, config: ModelArgs):
        self.config = config

        # store the original shape for unpermutation
        self.restore_shape = None
        # original token indices for permutated tokens
        self.sorted_indices = None
        # [n_experts, n_tokens]
        self.routing_map = None

    def token_permutation(
        self,
        tokens: torch.Tensor,  # [n_tokens, dim]
        routing_map: torch.Tensor,  # [n_tokens, n_experts]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.restore_shape = tokens.shape
        n_tokens, n_experts = routing_map.shape
        # [n_tokens, n_experts] => [n_experts]
        tokens_per_expert = routing_map.sum(dim=0)
        # [n_tokens, n_experts] => [n_experts, n_tokens]
        self.routing_map = routing_map.bool().T.contiguous()
        # [n_experts, n_tokens]
        token_indices = (
            torch.arange(n_tokens, device=routing_map.device)
            .unsqueeze(0)
            .expand(n_experts, -1)
        )
        # original token indices, sorted by expert idx
        self.sorted_indices = token_indices.masked_select(mask=self.routing_map)
        permuted_tokens = tokens.index_select(dim=0, index=self.sorted_indices)
        return permuted_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        permuted_tokens: torch.Tensor,  # [n_tokens * n_topk, dim]
        probs: torch.Tensor,  # [n_tokens, n_experts]
    ) -> torch.Tensor:
        _, hidden = permuted_tokens.shape
        # [n_experts, n_tokens] -> [n_permuted_tokens]
        permuted_probs = probs.T.contiguous().masked_select(mask=self.routing_map)
        # [n_permuted_tokens, dim]
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

        output = torch.zeros(
            self.restore_shape,
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        output.scatter_add_(
            dim=0,
            index=self.sorted_indices.unsqueeze(1).expand(-1, hidden),
            src=permuted_tokens,
        )

        return output.to(dtype=permuted_tokens.dtype)


# masked grouped gemm?
def sequential_gemm(input, weight, tokens_per_expert):
    num_tokens = input.shape[0]
    out_features = weight.shape[1]
    output = torch.zeros(
        num_tokens, out_features, dtype=input.dtype, device=input.device
    )

    cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
    # Insert zero at the begining for offset index's convenience
    zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
    cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

    for expert_num in range(weight.shape[0]):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        # tokens for this expert
        tokens = input[start:end]

        out = torch.matmul(tokens, weight[expert_num].T)
        output[start:end] = out
    return output


class ConditionalFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.intermediate_size)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )
        self.token_dispatcher = TokenDispatcher(config)

    def forward(
        self,
        x: Tensor,  # [n_tokens, dim]
        expert_indices: Tensor,  # [n_tokens, n_experts]
        expert_weights: Tensor,  # [n_tokens, n_experts]
    ) -> Tensor:
        # [n_tokens, dim] => [n_tokens * n_topk, dim] sorted by expert idx
        permuted_tokens, tokens_per_expert = self.token_dispatcher.token_permutation(
            x, expert_indices
        )

        # gemm for each expert
        x1 = sequential_gemm(permuted_tokens, self.w1, tokens_per_expert)
        x3 = sequential_gemm(permuted_tokens, self.w3, tokens_per_expert)

        up = F.silu(x1) * x3
        down = sequential_gemm(up, self.w2, tokens_per_expert)
        # weighted sum [n_tokens*n_topk, dim] => [n_tokens, dim]
        expert_outs = self.token_dispatcher.token_unpermutation(down, expert_weights)
        return expert_outs


class FeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MOEFeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(config)
        self.shared_ffn = FeedForward(
            config.dim, config.intermediate_size * config.num_shared_experts
        )
        self.dim = config.dim
        self.num_activated_experts = config.router_topk

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # [n_tokens, dim] => [n_tokens, n_experts]
        logits = self.gate(x)
        # [n_tokens, n_topk]
        expert_weights, expert_indices = torch.topk(
            logits, self.num_activated_experts, dim=-1
        )
        expert_weights = F.softmax(expert_weights, dim=-1)
        # convert expert_indices and expert_weights to dense mask
        routing_map = (
            torch.zeros_like(logits)
            .int()
            .scatter(dim=1, index=expert_indices, value=1)
            .bool()
        )
        probs = torch.zeros_like(logits).scatter(
            dim=1, index=expert_indices, src=expert_weights
        )

        expert_outs = self.cond_ffn(x, routing_map, probs)
        shared_outs = self.shared_ffn(x)
        return expert_outs + shared_outs


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    use_token_dispatcher = True
    n_tokens = 3
    args = ModelArgs()
    x = torch.randn((n_tokens, args.dim))
    moe = MOEFeedForward(args)
    print(moe(x))

    # scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
    # probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

    # probs, routing_map = self.router(hidden_states)
    # (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
    #     hidden_states, probs, routing_map
    # )
    # expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
    # output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
    # if self.use_shared_expert and not self.shared_expert_overlap:
    #     # if shared_expert_overlap is True, the expert calculation happens in
    #     # the token_dispatcher to overlap communications and computations
    #     output = output + self.shared_experts(hidden_states)
    # return output, mlp_bias

    # permuted_tokens, tokens_per_expert (local_expert)
