# ==============================================================================
# LAYER 3: PYTHON / PYTORCH - TRAINING HARNESS & LOSS COMPUTATION
# ==============================================================================

import torch
import torch.nn as nn
from .autograd import SheafDiffusionAutogradFunction

class SheafDiffusionLoss(nn.Module):
    """
    Multi-task loss module combining classification/regression loss
    with Dirichlet quadratic smoothing penalty to prevent oversmoothing.
    """
    def __init__(self, dirichlet_weight=0.01):
        super().__init__()
        self.dirichlet_weight = dirichlet_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, x, row_ptr, col_ind, rho):
        task_loss = self.ce_loss(logits, targets)

        # Compute Dirichlet energy penalty
        # 0.5 * sum_{(u,v)} ||rho_{uv} x_u - x_v||^2
        energy = torch.tensor(0.0, device=x.device)
        E = col_ind.shape[0]
        for e in range(E):
            u = int(col_ind[e])
            r = rho[e].view(4, 4)
            u_stalk = x[u].view(4, -1)
            contracted = torch.matmul(r, u_stalk).view(-1)
            # Find destination node from CSR
            # Quadratic penalty term
            diff = contracted - x[u]
            energy = energy + 0.5 * torch.sum(diff * diff)

        total_loss = task_loss + self.dirichlet_weight * energy
        return total_loss, task_loss, energy

class SheafDiffusionLayer(nn.Module):
    """
    PyTorch Neural Network Layer wrapping the SheafDiffusionAutogradFunction.
    """
    def __init__(self, tau=0.15, num_layers=10):
        super().__init__()
        self.tau = tau
        self.num_layers = num_layers

    def forward(self, x, row_ptr, col_ind, rho):
        return SheafDiffusionAutogradFunction.apply(
            x, row_ptr, col_ind, rho, self.tau, self.num_layers
        )
