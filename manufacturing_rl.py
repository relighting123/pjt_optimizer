"""
제조 스케줄링 도메인에서 Transformer 기반 PPO를 시연하는 단일 파일 예제입니다.
요구사항에 맞춘 커스텀 Gymnasium 환경, BC 사전학습, PPO 학습, 확장 환경 추론을 포함합니다.
"""

import os
import random
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed


# ---------------------------
# 전역 설정 값
# ---------------------------
SEED = 42
MAX_PROCESSES = 3
MAX_RESOURCES = 2
MAX_COUNT = 2
MAX_STEPS = 20
FEATURE_DIM = 4
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2


class PositionalEncoding(nn.Module):
    """Transformer 입력에 위치 정보를 부여하는 사인/코사인 기반 포지셔널 인코딩 클래스입니다."""

    def __init__(self, d_model: int, max_len: int) -> None:
        """지정된 모델 차원과 최대 시퀀스 길이에 맞는 포지셔널 인코딩을 초기화합니다."""
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 텐서에 포지셔널 인코딩을 더해 반환합니다."""
        return x + self.pe[:, : x.size(1)]


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """고정 길이 시퀀스를 TransformerEncoder로 처리하는 SB3용 Feature Extractor입니다."""

    def __init__(
        self,
        observation_space: spaces.Box,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
    ) -> None:
        """관측 텐서를 Transformer 입력으로 변환하는 프로젝션과 인코더를 구성합니다."""
        super().__init__(observation_space, features_dim=d_model)
        seq_len, feature_dim = observation_space.shape
        self.seq_len = seq_len
        self.projection = nn.Linear(feature_dim, d_model)
        self.positional = PositionalEncoding(d_model=d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """패딩 마스크를 적용한 Transformer 인코딩 후 평균 풀링 벡터를 생성합니다."""
        type_ids = observations[..., 0]
        padding_mask = type_ids.eq(-1.0)
        x = self.projection(observations)
        x = self.positional(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        x = x * valid_mask
        summed = x.sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp(min=1.0)
        return summed / denom


class CustomTransformerPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy에 Transformer 기반 Feature Extractor를 연결한 커스텀 정책입니다."""

    def __init__(self, *args, **kwargs) -> None:
        """Transformer Feature Extractor가 기본으로 사용되도록 초기화합니다."""
        kwargs.setdefault("features_extractor_class", TransformerFeatureExtractor)
        kwargs.setdefault(
            "features_extractor_kwargs",
            {"d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS},
        )
        super().__init__(*args, **kwargs)


class ManufacturingSchedulingEnv(gym.Env):
    """시간 슬롯 기반 장비 할당 문제를 단순화해 구현한 커스텀 Gymnasium 환경입니다."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_processes: int,
        max_resources: int,
        num_processes: int,
        num_resources: int,
        max_count: int,
        max_steps: int,
        seed: int = SEED,
    ) -> None:
        """환경의 최대/현재 공정 및 장비 수, 행동 공간, 관측 공간을 초기화합니다."""
        super().__init__()
        if num_processes > max_processes or num_resources > max_resources:
            raise ValueError("현재 공정/장비 수는 최대치 이하여야 합니다.")
        self.max_processes = max_processes
        self.max_resources = max_resources
        self.num_processes = num_processes
        self.num_resources = num_resources
        self.max_count = max_count
        self.max_steps = max_steps
        self.feature_dim = FEATURE_DIM
        self.max_elements = max_processes + max_resources
        self.action_space = spaces.MultiDiscrete(
            np.array([max_count] * (max_processes * max_resources), dtype=np.int64)
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1e6,
            shape=(self.max_elements, self.feature_dim),
            dtype=np.float32,
        )
        self._seed = seed
        self.np_random = None
        self.remaining_wip = np.zeros(self.num_processes, dtype=np.int64)
        self.process_time = np.ones(self.num_processes, dtype=np.float32)
        self.due_date = np.ones(self.num_processes, dtype=np.float32)
        self.total_target_wip = 1.0
        self.step_count = 0
        self.cumulative_busy = 0

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """무작위 초기 WIP/공정 시간을 설정하고 초기 관측을 반환합니다."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng(self._seed)
        self.process_time = self.np_random.integers(1, 4, size=self.num_processes)
        self.due_date = self.np_random.integers(5, 15, size=self.num_processes)
        self.remaining_wip = self.np_random.integers(3, 8, size=self.num_processes)
        self.total_target_wip = float(np.sum(self.remaining_wip))
        self.step_count = 0
        self.cumulative_busy = 0
        observation = self._build_observation()
        info = {"remaining_wip": self.remaining_wip.copy()}
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """행동에 따라 장비를 할당하고 보상 및 종료 여부를 계산합니다."""
        self.step_count += 1
        action_matrix = np.array(action, dtype=np.int64).reshape(
            self.max_processes, self.max_resources
        )
        allocations = np.zeros((self.num_processes, self.num_resources), dtype=np.int64)
        remaining_wip = self.remaining_wip.copy()
        throughput = 0
        used_resources = 0

        for r in range(self.num_resources):
            capacity = 1
            requests = [
                (p, int(action_matrix[p, r]))
                for p in range(self.num_processes)
                if action_matrix[p, r] > 0 and remaining_wip[p] > 0
            ]
            requests.sort(
                key=lambda x: (x[1], remaining_wip[x[0]]), reverse=True
            )
            for p, req in requests:
                if capacity <= 0:
                    break
                if remaining_wip[p] <= 0:
                    continue
                assign = min(req, remaining_wip[p], capacity)
                if assign > 0:
                    allocations[p, r] += assign
                    remaining_wip[p] -= assign
                    capacity -= assign
                    throughput += assign
            if capacity < 1:
                used_resources += 1

        self.remaining_wip = remaining_wip
        self.cumulative_busy += used_resources
        step_reward = throughput / self.total_target_wip
        terminated = bool(np.sum(self.remaining_wip) <= 0)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        utilization = (
            self.cumulative_busy / (self.num_resources * self.step_count)
            if self.step_count > 0
            else 0.0
        )
        final_reward = 0.0
        if terminated:
            final_reward = utilization * 10.0
        elif truncated:
            final_reward = -10.0
        reward = step_reward + final_reward

        observation = self._build_observation()
        info = {
            "throughput": throughput,
            "utilization": utilization,
            "allocations": allocations.copy(),
            "remaining_wip": self.remaining_wip.copy(),
        }
        return observation, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        """공정/장비 정보를 고정 길이 토큰 시퀀스로 구성해 반환합니다."""
        obs = np.zeros((self.max_elements, self.feature_dim), dtype=np.float32)
        for p in range(self.num_processes):
            obs[p] = np.array(
                [0.0, float(self.remaining_wip[p]), float(self.process_time[p]), float(self.due_date[p])],
                dtype=np.float32,
            )
        offset = self.num_processes
        for r in range(self.num_resources):
            obs[offset + r] = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        for i in range(offset + self.num_resources, self.max_elements):
            obs[i, 0] = -1.0
        return obs


def expert_policy_action(env: ManufacturingSchedulingEnv) -> np.ndarray:
    """잔여 WIP가 많은 공정에 장비를 우선 배분하는 규칙 기반 전문가 행동을 생성합니다."""
    action = np.zeros((env.max_processes, env.max_resources), dtype=np.int64)
    if env.num_processes == 0 or env.num_resources == 0:
        return action.flatten()
    remaining = env.remaining_wip.copy()
    order = np.argsort(-remaining)
    for r in range(env.num_resources):
        for p in order:
            if remaining[p] > 0:
                action[p, r] = 1
                break
    return action.flatten()


def generate_expert_data(
    env: ManufacturingSchedulingEnv, num_episodes: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """전문가 정책으로부터 관측/행동 쌍을 수집해 Behavior Cloning 데이터셋을 만듭니다."""
    observations = []
    actions = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = expert_policy_action(env)
            observations.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


def behavior_cloning_pretrain(
    model: PPO,
    expert_obs: np.ndarray,
    expert_actions: np.ndarray,
    batch_size: int = 64,
    epochs: int = 100,
) -> None:
    """PPO 학습 전 전문가 데이터를 사용해 정책망을 지도학습으로 사전 학습합니다."""
    policy = model.policy
    device = model.device
    dataset = TensorDataset(
        torch.tensor(expert_obs, dtype=torch.float32),
        torch.tensor(expert_actions, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy.train()
    for _ in range(epochs):
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            _, log_prob, _ = policy.evaluate_actions(obs_batch, act_batch)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    policy.eval()


def run_inference(model: PPO, env: ManufacturingSchedulingEnv) -> None:
    """확장된 환경에서 학습된 정책으로 추론을 수행하고 결과를 출력합니다."""
    obs, _ = env.reset()
    terminated = False
    truncated = False
    step_idx = 0
    print("=== 확장 환경 Inference 결과 ===")
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        allocations = info["allocations"]
        throughput = info["throughput"]
        utilization = info["utilization"]
        print(f"[슬롯 {step_idx}] 처리량={throughput} Utilization={utilization:.2f}")
        for r in range(env.num_resources):
            assigned = []
            for p in range(env.num_processes):
                qty = allocations[p, r]
                if qty > 0:
                    assigned.append(f"P{p}={qty}")
            assign_text = ", ".join(assigned) if assigned else "대기"
            print(f"  장비 R{r}: {assign_text}")
        step_idx += 1
    final_util = info["utilization"]
    status_text = "완료" if terminated else "미달성"
    print(f"최종 Utilization({status_text}): {final_util:.2f}")


def train_and_infer() -> None:
    """BC 사전학습, PPO 학습, 확장 환경 추론까지 전체 파이프라인을 실행합니다."""
    set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = ManufacturingSchedulingEnv(
        max_processes=MAX_PROCESSES,
        max_resources=MAX_RESOURCES,
        num_processes=2,
        num_resources=2,
        max_count=MAX_COUNT,
        max_steps=MAX_STEPS,
        seed=SEED,
    )

    model = PPO(
        CustomTransformerPolicy,
        train_env,
        verbose=1,
        seed=SEED,
        tensorboard_log="./tensorboard_logs",
    )

    expert_obs, expert_actions = generate_expert_data(train_env, num_episodes=50)
    behavior_cloning_pretrain(model, expert_obs, expert_actions, batch_size=64, epochs=100)

    model.learn(total_timesteps=5000, tb_log_name="ppo_manufacturing")

    inference_env = ManufacturingSchedulingEnv(
        max_processes=MAX_PROCESSES,
        max_resources=MAX_RESOURCES,
        num_processes=3,
        num_resources=2,
        max_count=MAX_COUNT,
        max_steps=MAX_STEPS,
        seed=SEED,
    )
    run_inference(model, inference_env)


if __name__ == "__main__":
    train_and_infer()
