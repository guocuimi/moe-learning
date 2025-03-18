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

        # store the hidden states shape for unpermutation
        self.hidden_states_shape = None
        self.reversed_input_permutation_mapping = None

    def token_permutation(
        self,
        hidden_states: torch.Tensor,  # [T, D]
        indices: torch.Tensor,  # [T, TopK]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.hidden_states_shape = hidden_states.shape
        # [..., T, D] -> [... * T, D]
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        # all expert idx [T*TopK]
        flatten_indices = indices.flatten()
        # sort expert idx stablely
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        # 0: [0, 1], 1: [2, 3], 2: [4, 5], ...
        token_indecies = sorted_indices // self.config.router_topk
        # tokens for each expert, sparse tensor
        permuted_tokens = hidden_states.index_select(dim=0, index=token_indecies)
        # record the permutation mapping for unpermutation
        self.reversed_input_permutation_mapping = sorted_indices

        # [n_experts]
        tokens_per_expert = torch.histc(
            flatten_indices,
            bins=self.config.num_experts,
            min=0,
            max=self.config.num_experts - 1,
        )
        return permuted_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        permuted_tokens: torch.Tensor,  # [n_tokens * n_topk, dim]
        probs: torch.Tensor,  # [n_tokens, n_topk]
    ) -> torch.Tensor:
        # num_unpermuted_tokens = n_tokens * n_topk
        num_unpermuted_tokens = probs.numel()
        # [n_permuted_tokens, dim]
        unpermuted_tokens = torch.zeros(
            (num_unpermuted_tokens, permuted_tokens.size(1)),
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        # sort back to original order, by token index
        unpermuted_tokens.index_copy_(
            dim=0, index=self.reversed_input_permutation_mapping, source=permuted_tokens
        )
        # [n_tokens * n_topk, dim] => [n_tokens, n_topk, dim]
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, self.config.router_topk, permuted_tokens.size(1)
        )
        # apply weights for each activated expert
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        # sum up the weighted tokens for each expert
        # [n_tokens, n_topk, dim] => [n_tokens, dim]
        unpermuted_tokens = unpermuted_tokens.sum(dim=1).type_as(permuted_tokens)
        # [n_tokens, dim] => [..., n_tokens, dim]
        output = unpermuted_tokens.view(self.hidden_states_shape)
        return output


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
        expert_indices: Tensor,  # [n_tokens, n_topk]
        expert_weights: Tensor,  # [n_tokens, n_topk]
    ) -> Tensor:
        if not use_token_dispatcher:
            # [n_tokens, n_topk] => [n_tokens, n_topk, intermediate_size, dim]
            w1_weights = self.w1[expert_indices]
            # [n_tokens, n_topk, intermediate_size, dim]
            w3_weights = self.w3[expert_indices]
            # [n_tokens, n_topk, dim, intermediate_size]
            w2_weights = self.w2[expert_indices]
            # w2(F.silu(w1(x)) * w3(x))
            # [n_tokens, dim] * [n_tokens, n_topk, intermediate_size, dim]
            #   => [n_tokens, n_topk, intermediate_size]
            x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
            # [n_tokens, n_topk, intermediate_size]
            x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
            # [n_tokens, n_topk, *intermediate_size] * [n_tokens, n_topk, dim, *intermediate_size]
            #   => [n_tokens, n_topk, dim]
            expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
            # [n_tokens, *n_topk, dim] * [n_tokens, *n_topk]
            #   => [n_tokens, dim]
            expert_outs = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        else:
            # [n_tokens, dim] => [n_tokens * n_topk, dim] sorted by expert idx
            permuted_tokens, tokens_per_expert = (
                self.token_dispatcher.token_permutation(x, expert_indices)
            )

            # gemm for each expert
            x1 = sequential_gemm(permuted_tokens, self.w1, tokens_per_expert)
            x3 = sequential_gemm(permuted_tokens, self.w3, tokens_per_expert)

            up = F.silu(x1) * x3
            down = sequential_gemm(up, self.w2, tokens_per_expert)
            # weighted sum [n_tokens*n_topk, dim] => [n_tokens, dim]
            expert_outs = self.token_dispatcher.token_unpermutation(
                down, expert_weights
            )
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
        expert_outs = self.cond_ffn(x, expert_indices, expert_weights)
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
