"""Small single-GPU Muon/AdamW split for the existing PyTorch image.

The Muon algorithm and quintic coefficients follow Keller Jordan's writeup
and the PyTorch Muon documentation; this is a local eager implementation.
https://github.com/KellerJordan/Muon
https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
"""

import math

import torch


def orthogonal_update(value, steps=5):
    original_dtype = value.dtype
    matrix = value.to(torch.bfloat16)
    transpose = matrix.shape[-2] > matrix.shape[-1]
    if transpose:
        matrix = matrix.mT
    matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = matrix @ matrix.mT
        correction = -4.775 * gram + 2.0315 * (gram @ gram)
        matrix = 3.4445 * matrix + correction @ matrix
    return (matrix.mT if transpose else matrix).to(original_dtype)


class MuonMatrices(torch.optim.Optimizer):
    def __init__(self, groups, lr, weight_decay):
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay, momentum=.95, row_blocks=1))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MuonMatrices does not use optimizer closures")
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.ndim != 2 or gradient.is_sparse:
                    raise ValueError("Muon requires dense 2D hidden weights")
                state = self.state[parameter]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1 - group["momentum"])
                update = gradient.lerp(buffer, group["momentum"])
                rows, columns = update.shape
                blocks = group["row_blocks"]
                if blocks > 1:
                    update = update.reshape(blocks, rows // blocks, columns)
                update = orthogonal_update(update).reshape_as(parameter)
                scale = math.sqrt(max(1., (rows / blocks) / columns))
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"] * scale)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, muon, adamw):
        self.muon, self.adamw = muon, adamw
        super().__init__(muon.param_groups + adamw.param_groups, {})

    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


def make_optimizer(model, name="adamw", lr=1e-4, weight_decay=.01, beta2=.999,
                   muon_lr=.01, fused=False, foreach=None):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                 betas=(.9, beta2), fused=fused, foreach=foreach)
    if name != "muon":
        raise ValueError("Unknown optimizer: " + name)
    matrices, qkv, other = [], [], []
    for key, parameter in model.named_parameters():
        # Recurrent gates and output classifiers stay on AdamW. Split packed
        # attention Q/K/V matrices for independent orthogonalization.
        auxiliary = any(part in key for part in ("high_update.", "update.weight_ih", "update.weight_hh",
                                                 "action_score.", "slot_gate."))
        if parameter.ndim != 2 or auxiliary:
            other.append(parameter)
        elif key.endswith("in_proj_weight"):
            qkv.append(parameter)
        else:
            matrices.append(parameter)
    groups = [{"params": matrices, "row_blocks": 1}, {"params": qkv, "row_blocks": 3}]
    muon = MuonMatrices([group for group in groups if group["params"]], muon_lr, weight_decay)
    adamw = torch.optim.AdamW(other, lr=lr, weight_decay=weight_decay, betas=(.9, beta2), fused=fused, foreach=foreach)
    return MuonAdamW(muon, adamw)
