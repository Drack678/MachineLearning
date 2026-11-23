"""Agente Double + Dueling DQN con proyección del estado oculto del Transformer (Fase 3)."""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.device import clip_grad_norm_safe


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        lm_hidden_dim: int,
        n_actions: int,
        context_proj_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.tabular_dim = tabular_dim
        self.lm_hidden_dim = lm_hidden_dim
        self.hparams = {
            "tabular_dim": tabular_dim,
            "lm_hidden_dim": lm_hidden_dim,
            "n_actions": n_actions,
            "context_proj_dim": context_proj_dim,
            "hidden_dim": hidden_dim,
        }

        self.context_proj: nn.Module | None
        if lm_hidden_dim > 0:
            self.context_proj = nn.Sequential(
                nn.Linear(lm_hidden_dim, context_proj_dim),
                nn.LayerNorm(context_proj_dim),
                nn.SiLU(),
            )
            trunk_in = tabular_dim + context_proj_dim
        else:
            self.context_proj = None
            trunk_in = tabular_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.context_proj is not None:
            tab = obs[:, : self.tabular_dim]
            lm_h = obs[:, self.tabular_dim :]
            fused = torch.cat([tab, self.context_proj(lm_h)], dim=-1)
        else:
            fused = obs
        h = self.trunk(fused)
        value = self.value_head(h)
        adv = self.adv_head(h)
        return value + adv - adv.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append((obs, int(action), float(reward), next_obs, float(terminated)))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, terminated = zip(*batch, strict=True)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(terminated, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        tabular_dim: int,
        lm_hidden_dim: int,
        n_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        buffer_size: int = 40_000,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.q = DuelingQNetwork(tabular_dim, lm_hidden_dim, n_actions).to(self.device)
        self.target_q = DuelingQNetwork(tabular_dim, lm_hidden_dim, n_actions).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optimizer = torch.optim.AdamW(self.q.parameters(), lr=lr, weight_decay=weight_decay)
        self.buffer = ReplayBuffer(buffer_size)

    @torch.inference_mode()
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        self.q.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q(obs_t)
        return int(q_values.argmax(dim=-1).item())

    def update(self) -> float:
        self.q.train()
        obs, actions, rewards, next_obs, terminated = self.buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        term_t = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)
        q_values = self.q(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target_q(next_obs_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * (1.0 - term_t) * next_q

        loss = nn.functional.smooth_l1_loss(q_values, target)
        loss.backward()
        _, is_finite = clip_grad_norm_safe(self.q.parameters(), max_norm=1.0)
        if is_finite:
            self.optimizer.step()
            self.soft_update_target()
        return float(loss.item())

    @torch.no_grad()
    def soft_update_target(self) -> None:
        for p_target, p_online in zip(self.target_q.parameters(), self.q.parameters(), strict=True):
            p_target.data.mul_(1.0 - self.tau).add_(p_online.data, alpha=self.tau)

    def update_target(self) -> None:
        self.target_q.load_state_dict(self.q.state_dict())
