"""
Muon optimizer - Momentum Orthogonalized by Newton-Schulz.

Adapted from modded-nanogpt:
https://github.com/KellerJordan/modded-nanogpt

Uses Polar Express Sign Method for orthogonalization:
https://arxiv.org/pdf/2505.16932
"""

from functools import partial
from typing import List

import torch
from torch import Tensor
import torch.distributed as dist


# Polar Express coefficients (computed for num_iters=5, safety_factor=2e-2, cushion=2)
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    """Fused Muon step: momentum -> polar_express -> variance_reduction -> update."""

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express orthogonalization
    X = g.bfloat16()
    if g.size(-2) > g.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if g.size(-2) > g.size(-1):
        X = X.mT
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz.

    Should only be used for 2D parameters (attention/MLP weights).
    Use AdamW for embeddings and other 1D parameters.

    Args:
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        ns_steps: Number of Newton-Schulz iteration steps (default: 5)
        beta2: Second moment decay rate (default: 0.95)
        weight_decay: Cautious weight decay (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"

        # Group by shape for batched updates
        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            param_groups.append(dict(params=group_params))

        super().__init__(param_groups, defaults)

        # 0-D CPU tensors for torch.compile compatibility
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: List[Tensor] = group["params"]
            if not params:
                continue

            state = self.state[params[0]]
            num_params = len(params)
            shape, device, dtype = params[0].shape, params[0].device, params[0].dtype

            # Initialize buffers
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros(
                    num_params, *shape, dtype=dtype, device=device
                )
            momentum_buffer = state["momentum_buffer"]

            if "second_momentum_buffer" not in state:
                if shape[-2] >= shape[-1]:
                    state["second_momentum_buffer"] = torch.zeros(
                        num_params, shape[-2], 1, dtype=dtype, device=device
                    )
                else:
                    state["second_momentum_buffer"] = torch.zeros(
                        num_params, 1, shape[-1], dtype=dtype, device=device
                    )
            second_momentum_buffer = state["second_momentum_buffer"]
            red_dim = -1 if shape[-2] >= shape[-1] else -2

            # Stack grads and params
            stacked_grads = torch.stack([p.grad for p in params])
            stacked_params = torch.stack(params)

            # Fill 0-D tensors
            self._momentum_t.fill_(group["momentum"])
            self._beta2_t.fill_(group["beta2"] if group["beta2"] else 0.0)
            self._lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._wd_t.fill_(group["weight_decay"])

            # Fused update
            muon_step_fused(
                stacked_grads,
                stacked_params,
                momentum_buffer,
                second_momentum_buffer,
                self._momentum_t,
                self._lr_t,
                self._wd_t,
                self._beta2_t,
                group["ns_steps"],
                red_dim,
            )

            # Copy back to original params
            torch._foreach_copy_(params, list(stacked_params.unbind(0)))


class DistMuon(torch.optim.Optimizer):
    """Distributed version of Muon optimizer."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            chunk_size = (len(group_params) + world_size - 1) // world_size
            if rank == 0:
                print(
                    f"Muon: {len(group_params)} params of shape {shape}, chunk={chunk_size}"
                )
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        super().__init__(param_groups, defaults)

        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        group_infos = []
        for group in self.param_groups:
            params: List[Tensor] = group["params"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * world_size
            shape = params[0].shape
            device, dtype = params[0].device, params[0].dtype

            grad_stack = torch.stack([p.grad for p in params])
            stacked_grads = torch.empty(
                padded_num_params, *shape, dtype=dtype, device=device
            )
            stacked_grads[: len(params)].copy_(grad_stack)
            if len(params) < padded_num_params:
                stacked_grads[len(params) :].zero_()

            grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()

            group_infos.append(
                dict(
                    grad_chunk=grad_chunk,
                    reduce_future=reduce_future,
                    stacked_grads=stacked_grads,
                )
            )

        all_gather_futures = []
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = group["params"]
            chunk_size = group["chunk_size"]
            shape = params[0].shape
            device, dtype = params[0].device, params[0].dtype
            grad_chunk = info["grad_chunk"]

            start_idx = rank * chunk_size
            num_owned = min(chunk_size, max(0, len(params) - start_idx))

            state = self.state[params[0]]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros(
                    chunk_size, *shape, dtype=dtype, device=device
                )
            momentum_buffer = state["momentum_buffer"]

            if "second_momentum_buffer" not in state:
                if shape[-2] >= shape[-1]:
                    state["second_momentum_buffer"] = torch.zeros(
                        chunk_size, shape[-2], 1, dtype=dtype, device=device
                    )
                else:
                    state["second_momentum_buffer"] = torch.zeros(
                        chunk_size, 1, shape[-1], dtype=dtype, device=device
                    )
            second_momentum_buffer = state["second_momentum_buffer"]
            red_dim = -1 if shape[-2] >= shape[-1] else -2

            updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

            if num_owned > 0:
                owned_params = [params[start_idx + i] for i in range(num_owned)]
                stacked_owned_params = torch.stack(owned_params)

                owned_grads = grad_chunk[:num_owned]
                owned_momentum = momentum_buffer[:num_owned]
                owned_second_momentum = second_momentum_buffer[:num_owned]

                self._momentum_t.fill_(group["momentum"])
                self._beta2_t.fill_(group["beta2"] if group["beta2"] else 0.0)
                self._lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
                self._wd_t.fill_(group["weight_decay"])

                muon_step_fused(
                    owned_grads,
                    stacked_owned_params,
                    owned_momentum,
                    owned_second_momentum,
                    self._momentum_t,
                    self._lr_t,
                    self._wd_t,
                    self._beta2_t,
                    group["ns_steps"],
                    red_dim,
                )

                updated_params[:num_owned].copy_(stacked_owned_params)

            if num_owned < chunk_size:
                updated_params[num_owned:].zero_()

            stacked_params = info["stacked_grads"]
            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
            ).get_future()

            all_gather_futures.append(
                dict(
                    gather_future=gather_future,
                    stacked_params=stacked_params,
                    params=params,
                )
            )

        for info in all_gather_futures:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            params = info["params"]
            torch._foreach_copy_(params, list(stacked_params[: len(params)].unbind(0)))
