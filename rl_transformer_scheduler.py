"""
제조 스케줄링(RL/Transformer) 데모 - 단일 파일 실행 스크립트

요구사항 요약
- Gymnasium 커스텀 환경: 시간 슬롯별 장비 할당을 통해 WIP 처리량 및 Utilization 최대화
- Observation: (Max_Elements, Feature_Dim) 고정 텐서 (Padding 포함)
- Action: MultiDiscrete([Max_Count] * (Max_Processes * Max_Resources)) 고정 크기, 범위 밖 무시(Masking)
- Reward:
  - Step Reward: (현재 슬롯 처리량 / 전체 목표 WIP)
  - Final Reward: 모든 WIP 완료 시 (총 가동률 * 10), 미달성 시 -10
- 모델: Stable-Baselines3 PPO + TransformerEncoder feature extractor
- Expert(휴리스틱) 데이터 생성 + Behavior Cloning 100 epoch 사전학습 후 PPO 학습
- 1단계: 공정2/장비2로 5,000 step 학습(Tensorboard 로그)
- 2단계: 동일 모델로 공정3/장비2 환경에서 추론 및 사람이 읽기 쉬운 출력

주의
- 이 파일은 "코드가 한 파일"이라는 요구사항을 만족합니다.
- 실행 전 의존성: numpy, torch, gymnasium, stable-baselines3 설치 필요
  예) pip install numpy torch gymnasium stable-baselines3 tensorboard
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed


# =========================
# 0) 전역 시드 고정 (요구사항: 42)
# =========================
SEED = 42


# =========================
# 1) 환경 정의: ManufacturingSchedulingEnv
# =========================
@dataclass
class EnvConfig:
    """환경 파라미터 묶음 (가독성과 재사용성을 위해 dataclass 사용)."""

    # 전역 최대 차원(모델/공간 차원을 고정하기 위한 상수)
    max_processes: int = 3
    max_resources: int = 2

    # 실제로 사용하는 공정/장비 수 (가변)
    num_processes: int = 2
    num_resources: int = 2

    # 액션 각 차원의 최대 카운트(= 각 공정-장비 페어에 할당 가능한 최대 수량)
    # Gymnasium MultiDiscrete의 nvec 특성상 실제 액션 값 범위는 [0, max_count-1] 입니다.
    # 즉, "할당 가능한 최대 수량"으로 해석할 때는 (max_count-1)을 최대치로 봅니다.
    max_count: int = 8

    # 에피소드 길이(시간 슬롯 수) 제한
    horizon: int = 20

    # 장비 1대가 슬롯 당 보유한 작업 시간(정규화 시간)
    slot_time: float = 1.0

    # WIP / 공정 파라미터 범위(초기화 랜덤)
    wip_low: int = 8
    wip_high: int = 20
    proc_time_low: float = 0.2
    proc_time_high: float = 0.6
    due_low: float = 5.0
    due_high: float = 25.0


class ManufacturingSchedulingEnv(gym.Env):
    """
    제조 스케줄링용 Gymnasium 커스텀 환경.

    - 목적: 시간 슬롯 단위로 공정(WIP)을 장비에 배치하여 처리량/가동률을 높임
    - Observation: (Max_Elements, Feature_Dim) 고정 텐서
      - Max_Elements = Max_Processes + Max_Resources
      - 각 토큰 Feature = [Type_ID, Current_Qty, Process_Time, Due_Date]
      - padding 토큰(Type_ID=0)은 마스킹 대상으로 처리
      - 공정 토큰(Type_ID=1), 장비 토큰(Type_ID=2)
    - Action: MultiDiscrete([Max_Count] * (Max_Processes * Max_Resources))
      - 인덱스 (p, r) -> a[p * Max_Resources + r]
      - 실제 공정/장비 범위를 벗어난 액션은 무시(Masking)
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg

        self.max_processes = cfg.max_processes
        self.max_resources = cfg.max_resources
        self.max_elements = self.max_processes + self.max_resources
        self.feature_dim = 4

        # 관측: 고정 크기 행렬 (Max_Elements, Feature_Dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_elements, self.feature_dim),
            dtype=np.float32,
        )

        # 액션: (Max_Processes * Max_Resources) 길이의 MultiDiscrete
        self.action_dim = self.max_processes * self.max_resources
        self.action_space = spaces.MultiDiscrete([cfg.max_count] * self.action_dim)

        # 내부 상태(초기화는 reset에서 수행)
        self._t = 0  # 현재 시간 슬롯 인덱스
        self._remaining_wip = np.zeros(self.max_processes, dtype=np.int32)  # 공정별 잔여 WIP
        self._process_time = np.ones(self.max_processes, dtype=np.float32)  # 공정별 단위 처리시간
        self._due_date = np.ones(self.max_processes, dtype=np.float32)  # 공정별 due date(상대값)

        # 누적 통계(가동률 계산용)
        self._cum_used_time = np.zeros(self.max_resources, dtype=np.float32)
        self._cum_available_time = 0.0

        # 목표 WIP(초기 총량) - step reward 분모
        self._target_total_wip = 1.0

    def seed(self, seed: Optional[int] = None) -> None:
        """(호환용) 환경 내부 랜덤 시드 설정."""
        if seed is None:
            seed = SEED
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        에피소드 초기화.
        - 실제 유효 공정/장비 수(cfg.num_processes/num_resources)는 유지
        - 나머지 차원은 padding으로 둠(관측에서 Type_ID=0으로 표현)
        """
        super().reset(seed=seed)
        if seed is None:
            seed = SEED
        self.seed(seed)

        self._t = 0
        self._cum_used_time[:] = 0.0
        self._cum_available_time = 0.0

        # 유효 공정 영역만 랜덤 초기화, 나머지는 padding 값 유지
        self._remaining_wip[:] = 0
        self._process_time[:] = 0.0
        self._due_date[:] = 0.0

        for p in range(self.cfg.num_processes):
            self._remaining_wip[p] = np.random.randint(self.cfg.wip_low, self.cfg.wip_high + 1)
            self._process_time[p] = np.random.uniform(self.cfg.proc_time_low, self.cfg.proc_time_high)
            # due date는 단순히 상대 기한(슬롯 기준)으로 가정
            self._due_date[p] = np.random.uniform(self.cfg.due_low, self.cfg.due_high)

        self._target_total_wip = float(np.sum(self._remaining_wip[: self.cfg.num_processes]))
        if self._target_total_wip <= 0:
            self._target_total_wip = 1.0

        obs = self._build_observation()
        info: Dict[str, Any] = {"t": self._t}
        return obs, info

    def _build_observation(self) -> np.ndarray:
        """
        고정 크기 관측 생성.
        - [0 : max_processes) 구간: 공정 토큰 (Type_ID=1 또는 padding=0)
        - [max_processes : max_elements) 구간: 장비 토큰 (Type_ID=2 또는 padding=0)
        """
        obs = np.zeros((self.max_elements, self.feature_dim), dtype=np.float32)

        # 공정 토큰
        for p in range(self.max_processes):
            if p < self.cfg.num_processes:
                obs[p, 0] = 1.0  # Type_ID=1 (process)
                obs[p, 1] = float(self._remaining_wip[p])
                obs[p, 2] = float(self._process_time[p])
                obs[p, 3] = float(self._due_date[p])
            else:
                # padding: Type_ID=0 유지
                pass

        # 장비 토큰
        base = self.max_processes
        for r in range(self.max_resources):
            idx = base + r
            if r < self.cfg.num_resources:
                obs[idx, 0] = 2.0  # Type_ID=2 (resource)
                # Current_Qty는 장비의 "가용성" 같은 단순 신호로 1.0 고정(예시)
                obs[idx, 1] = 1.0
                obs[idx, 2] = 0.0
                obs[idx, 3] = 0.0
            else:
                # padding
                pass

        return obs

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """
        1차원 MultiDiscrete 액션을 (max_processes, max_resources) 매트릭스로 변환.
        """
        a = np.asarray(action, dtype=np.int32).reshape(self.max_processes, self.max_resources)
        return a

    def step(self, action):
        """
        시간 슬롯 1 step 진행.
        - 무효 공정/장비 인덱스에 대한 액션은 내부적으로 0으로 간주 (masking)
        - 처리 가능한 수량은 (공정 잔여 WIP, 장비 슬롯 시간, 공정 단위 처리시간, max_count) 제한을 모두 만족
        """
        self._t += 1
        action_matrix = self._decode_action(action)

        # 무효 영역 마스킹: 실제 공정/장비 범위를 벗어나면 0으로 무시
        masked_action = np.zeros_like(action_matrix, dtype=np.int32)
        masked_action[: self.cfg.num_processes, : self.cfg.num_resources] = action_matrix[
            : self.cfg.num_processes, : self.cfg.num_resources
        ]

        # 슬롯 당 장비 가용 시간
        resource_time_left = np.full(self.max_resources, self.cfg.slot_time, dtype=np.float32)
        used_time = np.zeros(self.max_resources, dtype=np.float32)

        # 실제 처리량(이번 슬롯에서 처리된 WIP 수량)
        throughput = 0
        executed = np.zeros_like(masked_action, dtype=np.int32)

        # 공정-장비 배분을 순회하며 처리 (간단한 시뮬레이션)
        for r in range(self.cfg.num_resources):
            for p in range(self.cfg.num_processes):
                req_qty = int(masked_action[p, r])
                if req_qty <= 0:
                    continue
                if self._remaining_wip[p] <= 0:
                    continue

                pt = float(self._process_time[p])
                if pt <= 0:
                    continue

                # 남은 시간으로 처리 가능한 최대 수량(정수)
                max_by_time = int(math.floor(resource_time_left[r] / pt + 1e-9))
                if max_by_time <= 0:
                    continue

                # MultiDiscrete 상한 고려: 실제 액션 최대값은 (max_count-1)
                max_action_value = int(self.cfg.max_count) - 1
                can_do = min(req_qty, max_action_value, int(self._remaining_wip[p]), max_by_time)
                if can_do <= 0:
                    continue

                executed[p, r] = can_do
                self._remaining_wip[p] -= can_do
                throughput += can_do

                t_used = float(can_do) * pt
                resource_time_left[r] -= t_used
                used_time[r] += t_used

        # 가동률 누적(이번 슬롯의 가용 시간도 누적)
        self._cum_used_time[: self.cfg.num_resources] += used_time[: self.cfg.num_resources]
        self._cum_available_time += float(self.cfg.num_resources) * float(self.cfg.slot_time)

        # Step reward: 현재 슬롯 처리량 / 전체 목표 WIP
        step_reward = float(throughput) / float(self._target_total_wip)

        # 종료 조건
        done_all = bool(np.sum(self._remaining_wip[: self.cfg.num_processes]) <= 0)
        truncated = bool(self._t >= self.cfg.horizon and not done_all)
        terminated = bool(done_all)

        # Final reward: 성공 시 총 가동률*10, 실패 시 -10
        final_reward = 0.0
        if terminated or truncated:
            util = self.get_utilization()
            if terminated:
                final_reward = util * 10.0
            else:
                final_reward = -10.0

        reward = step_reward + final_reward
        obs = self._build_observation()

        info: Dict[str, Any] = {
            "t": self._t,
            "throughput": throughput,
            "step_reward": step_reward,
            "final_reward": final_reward,
            "remaining_wip": self._remaining_wip.copy(),
            "executed_matrix": executed.copy(),
            "masked_action_matrix": masked_action.copy(),
            "used_time": used_time.copy(),
            "utilization": self.get_utilization(),
            "terminated": terminated,
            "truncated": truncated,
        }
        return obs, reward, terminated, truncated, info

    def get_utilization(self) -> float:
        """
        현재까지의 총 가동률(utilization) 계산.
        - 총 사용시간 / 총 가용시간
        """
        if self._cum_available_time <= 1e-9:
            return 0.0
        return float(np.sum(self._cum_used_time[: self.cfg.num_resources]) / self._cum_available_time)


# =========================
# 2) Transformer 기반 Feature Extractor
# =========================
class PositionalEncoding(nn.Module):
    """
    표준 사인/코사인 Positional Encoding.
    - Transformer에 순서 정보를 제공하기 위해 사용
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class TransformerTokenExtractor(BaseFeaturesExtractor):
    """
    SB3에 연결되는 Transformer 기반 Feature Extractor.

    요구사항:
    - torch.nn.TransformerEncoder(d_model=64, nhead=4, layers=2)
    - 입력 Linear Projection + Positional Encoding
    - Padding Mask(src_key_padding_mask) 로직 포함 (Type_ID=0인 토큰을 padding으로 간주)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        # features_dim은 Transformer 출력을 pooling한 벡터 차원
        super().__init__(observation_space, features_dim=d_model)
        self.d_model = d_model

        # 관측 shape: (max_elements, feature_dim)
        max_elements, feature_dim = observation_space.shape

        # 입력 feature(4차원)를 d_model(64)로 투영
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_elements)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (batch, seq, dim)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # pooling 후 안정화를 위한 LayerNorm
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch, max_elements, feature_dim)
        return: (batch, d_model)
        """
        # padding mask 생성: Type_ID==0이면 padding(True)
        # Type_ID는 feature[0]
        type_ids = observations[..., 0]
        src_key_padding_mask = type_ids.eq(0.0)  # (batch, seq)

        x = self.input_proj(observations)  # (batch, seq, d_model)
        x = self.pos_enc(x)

        # Transformer 인코딩 (padding 토큰은 attention에서 제외)
        z = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # padding을 제외한 mean pooling
        # mask(False)=유효 토큰, mask(True)=padding 토큰
        valid_mask = (~src_key_padding_mask).unsqueeze(-1)  # (batch, seq, 1)
        z_masked = z * valid_mask
        denom = valid_mask.sum(dim=1).clamp(min=1.0)
        pooled = z_masked.sum(dim=1) / denom

        return self.out_norm(pooled)


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """
    SB3용 Custom ActorCriticPolicy.

    요구사항:
    - "SB3의 Custom ActorCriticPolicy를 사용하여 위 Extractor 연결"
      => 기본 MlpPolicy 문자열 대신, ActorCriticPolicy를 상속한 커스텀 클래스를 직접 사용합니다.
    - 내부 feature extractor로 TransformerTokenExtractor를 사용합니다.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        # 기본적으로 Transformer 기반 extractor를 연결하고,
        # pi/vf head는 작은 MLP로 구성합니다(필요 시 net_arch 변경 가능).
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=TransformerTokenExtractor,
            features_extractor_kwargs=dict(d_model=64, nhead=4, num_layers=2, dropout=0.1),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            *args,
            **kwargs,
        )


# =========================
# 3) Expert 데이터 생성 + BC(Behavior Cloning) 사전학습
# =========================
def expert_policy_action(env: ManufacturingSchedulingEnv) -> np.ndarray:
    """
    Expert 규칙 기반 액션 생성.
    - 예시 규칙: 잔여 WIP가 가장 많은 공정을 선택하여, 가능한 한 모든 장비에 우선 배분

    반환:
    - MultiDiscrete 액션 벡터(shape=(max_processes*max_resources,))
    """
    cfg = env.cfg
    max_p, max_r = env.max_processes, env.max_resources
    act = np.zeros((max_p, max_r), dtype=np.int32)

    # 유효 공정 중 잔여 WIP 최대 공정 선택
    rem = env._remaining_wip[: cfg.num_processes]
    if rem.size == 0:
        return act.reshape(-1)
    p_star = int(np.argmax(rem))
    if rem[p_star] <= 0:
        return act.reshape(-1)

    # 선택된 공정을 각 장비에 최대한 배분(시간/상한 고려는 env.step에서 최종 적용됨)
    # (MultiDiscrete 값 범위를 지키기 위해 최대값은 max_count-1)
    max_action_value = int(cfg.max_count) - 1
    for r in range(cfg.num_resources):
        act[p_star, r] = max_action_value

    return act.reshape(-1)


def collect_expert_dataset(
    env: ManufacturingSchedulingEnv,
    num_episodes: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expert 정책으로 roll-out하여 (obs, action) 지도학습 데이터 생성.

    반환:
    - obs_arr: (N, max_elements, feature_dim)
    - act_arr: (N, action_dim)
    """
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        while not done:
            act = expert_policy_action(env)
            obs_list.append(obs.copy())
            act_list.append(act.copy())

            obs, _, terminated, truncated, _ = env.step(act)
            done = bool(terminated or truncated)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.int64)
    return obs_arr, act_arr


def behavior_cloning_pretrain(
    model: PPO,
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
) -> None:
    """
    PPO 학습 시작 전, Expert 데이터로 Policy Network를 Supervised Learning(BC) 사전학습.

    핵심 포인트:
    - SB3 policy의 `evaluate_actions()`를 사용하면 MultiDiscrete 환경에서도 log_prob 계산 가능
    - 손실: NLL = -log_prob(expert_action)
    """
    policy = model.policy
    policy.to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    n = obs_arr.shape[0]
    indices = np.arange(n)

    for ep in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_b = torch.as_tensor(obs_arr[batch_idx], dtype=torch.float32, device=device)
            act_b = torch.as_tensor(act_arr[batch_idx], dtype=torch.long, device=device)

            # evaluate_actions: (values, log_prob, entropy)
            _, log_prob, entropy = policy.evaluate_actions(obs_b, act_b)

            # Behavior Cloning 손실(음의 로그우도) + 약한 엔트로피 정규화(과도한 확신 방지)
            loss = (-log_prob).mean() - 0.001 * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()


# =========================
# 4) 학습/추론 루틴
# =========================
def make_env(cfg: EnvConfig) -> ManufacturingSchedulingEnv:
    """SB3 DummyVecEnv에서 사용하기 위한 env 생성 함수."""
    env = ManufacturingSchedulingEnv(cfg)
    env.reset(seed=SEED)
    return env


def print_inference_report(step_records: List[Dict[str, Any]], final_util: float) -> None:
    """
    사람이 읽기 쉬운 형태로 시간 슬롯별 할당 결과와 최종 Utilization 출력.
    """
    print("\n==============================")
    print("추론 결과 리포트 (시간 슬롯별)")
    print("==============================")
    for rec in step_records:
        t = rec["t"]
        executed = rec["executed_matrix"]
        masked_action = rec["masked_action_matrix"]
        throughput = rec["throughput"]
        util = rec["utilization"]
        remaining = rec["remaining_wip"]

        print(f"\n[슬롯 {t}] 처리량={throughput}, 누적 Utilization={util:.3f}")
        print("- 할당(Action, 마스킹 후) 매트릭스 [process x resource]:")
        print(masked_action)
        print("- 실제 처리(Executed) 매트릭스 [process x resource]:")
        print(executed)
        print(f"- 잔여 WIP: {remaining}")
    print("\n==============================")
    print(f"최종 Utilization: {final_util:.3f}")
    print("==============================\n")


def main() -> None:
    """
    전체 실행 엔트리포인트.
    - 1단계: (공정2, 장비2) 환경에서 BC 사전학습 + PPO 5,000 step 학습 (Tensorboard 로깅)
    - 2단계: 동일 모델을 (공정3, 장비2) 환경에 적용하여 추론 및 결과 출력
    """
    # --- 시드 고정 ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # --- 1단계 환경(공정2/장비2), 단 모델 차원은 max_processes=3/max_resources=2로 고정 ---
    cfg_train = EnvConfig(num_processes=2, num_resources=2, max_processes=3, max_resources=2)
    env_train_raw = make_env(cfg_train)
    env_train = DummyVecEnv([lambda: make_env(cfg_train)])

    # --- PPO 모델 생성: TransformerExtractor + PPO ---
    # Tensorboard 로그 폴더
    tb_log_dir = os.path.join(os.getcwd(), "tb_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    model = PPO(
        policy=TransformerActorCriticPolicy,  # 요구사항: Custom ActorCriticPolicy 사용
        env=env_train,
        seed=SEED,
        verbose=1,
        tensorboard_log=tb_log_dir,
        device=device,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )

    # --- Expert 데이터 생성 + BC 사전학습(100 epoch) ---
    obs_arr, act_arr = collect_expert_dataset(env_train_raw, num_episodes=60)
    behavior_cloning_pretrain(model, obs_arr, act_arr, epochs=100, batch_size=64, lr=3e-4, device=device)

    # --- PPO 학습(5,000 step), Tensorboard logging ---
    model.learn(total_timesteps=5_000, tb_log_name="ppo_transformer_sched")

    # --- 2단계: 확장 환경(공정3/장비2)에서 동일 모델로 추론 ---
    cfg_test = EnvConfig(num_processes=3, num_resources=2, max_processes=3, max_resources=2)
    env_test = make_env(cfg_test)

    obs, info = env_test.reset(seed=SEED)
    done = False
    step_records: List[Dict[str, Any]] = []

    while not done:
        # SB3는 VecEnv가 아니어도 predict 가능하지만, obs shape에 주의
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        step_records.append(info)
        done = bool(terminated or truncated)

    final_util = env_test.get_utilization()
    print_inference_report(step_records, final_util)


if __name__ == "__main__":
    main()

