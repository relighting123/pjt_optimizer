"""
제조 스케줄링 RL/Transformer 시스템
=====================================
목적: 시간 슬롯별 장비 할당을 통해 WIP(Work In Progress) 처리량 및 Utilization 극대화
알고리즘: PPO (Proximal Policy Optimization) + Transformer Feature Extractor
사전학습: Behavior Cloning (Expert Knowledge 기반)

저자: AI Assistant
날짜: 2026-01-21
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 전역 상수 및 시드 설정
# ==============================================================================
RANDOM_SEED = 42  # 재현성을 위한 고정 시드
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 환경 최대 크기 설정 (패딩을 위한 고정 차원)
MAX_PROCESSES = 5      # 최대 공정 수
MAX_RESOURCES = 5      # 최대 장비(리소스) 수
MAX_ELEMENTS = MAX_PROCESSES + MAX_RESOURCES  # Observation 토큰 최대 개수
FEATURE_DIM = 4        # 각 토큰의 특성 차원: [Type_ID, Current_Qty, Process_Time, Due_Date]
MAX_COUNT = 10         # 각 (공정, 장비) 조합에 할당 가능한 최대 수량


# ==============================================================================
# Gymnasium Custom Environment: 제조 스케줄링 환경
# ==============================================================================
class ManufacturingSchedulingEnv(gym.Env):
    """
    제조 스케줄링을 위한 커스텀 Gymnasium 환경
    
    환경 설명:
    -----------
    - 여러 공정(Process)과 장비(Resource)가 존재
    - 각 시간 슬롯에서 장비를 공정에 할당하여 WIP를 처리
    - 목표: 모든 WIP를 처리하면서 장비 가동률(Utilization) 극대화
    
    Observation Space:
    ------------------
    - 형태: (MAX_ELEMENTS, FEATURE_DIM)
    - 각 토큰: [Type_ID, Current_Qty, Process_Time, Due_Date]
      * Type_ID: 0=패딩, 1=공정, 2=장비
      * Current_Qty: 현재 WIP 수량 또는 장비 용량
      * Process_Time: 처리 시간 또는 장비 처리 속도
      * Due_Date: 납기일 또는 장비 가용 시간
    
    Action Space:
    -------------
    - MultiDiscrete([MAX_COUNT] * (MAX_PROCESSES * MAX_RESOURCES))
    - 각 액션 값: 해당 (공정, 장비) 조합에 할당할 수량
    - 유효하지 않은 (공정, 장비) 조합의 액션은 내부적으로 무시됨
    
    Reward:
    -------
    - Step Reward: (현재 슬롯 처리량 / 전체 목표 WIP)
    - Final Reward: 모든 WIP 완료 시 (총 가동률 * 10), 미달성 시 -10 패널티
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        num_processes: int = 2,
        num_resources: int = 2,
        initial_wip: Optional[List[int]] = None,
        max_time_slots: int = 20,
        render_mode: Optional[str] = None
    ):
        """
        환경 초기화
        
        매개변수:
        ---------
        num_processes : int
            실제 사용할 공정 수 (MAX_PROCESSES 이하)
        num_resources : int
            실제 사용할 장비 수 (MAX_RESOURCES 이하)
        initial_wip : List[int], optional
            각 공정의 초기 WIP 수량. None이면 랜덤 생성
        max_time_slots : int
            최대 시간 슬롯 수 (에피소드 최대 길이)
        render_mode : str, optional
            렌더링 모드
        """
        super().__init__()
        
        # 환경 설정 검증
        assert num_processes <= MAX_PROCESSES, f"공정 수는 {MAX_PROCESSES} 이하여야 합니다."
        assert num_resources <= MAX_RESOURCES, f"장비 수는 {MAX_RESOURCES} 이하여야 합니다."
        
        self.num_processes = num_processes  # 실제 공정 수
        self.num_resources = num_resources  # 실제 장비 수
        self.max_time_slots = max_time_slots  # 최대 시간 슬롯
        self.render_mode = render_mode
        
        # 초기 WIP 설정
        if initial_wip is not None:
            assert len(initial_wip) == num_processes, "초기 WIP 길이가 공정 수와 일치해야 합니다."
            self.initial_wip = np.array(initial_wip, dtype=np.float32)
        else:
            self.initial_wip = None  # reset()에서 랜덤 생성
        
        # Observation Space: 고정 크기 (MAX_ELEMENTS, FEATURE_DIM)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(MAX_ELEMENTS, FEATURE_DIM),
            dtype=np.float32
        )
        
        # Action Space: MultiDiscrete - 고정 크기 (MAX_PROCESSES * MAX_RESOURCES)
        self.action_space = spaces.MultiDiscrete(
            [MAX_COUNT] * (MAX_PROCESSES * MAX_RESOURCES)
        )
        
        # 내부 상태 변수 초기화
        self.current_wip = None           # 현재 WIP 수량
        self.resource_capacity = None      # 장비별 처리 용량
        self.process_times = None          # 공정별 처리 시간
        self.due_dates = None              # 공정별 납기일
        self.current_slot = 0              # 현재 시간 슬롯
        self.total_initial_wip = 0         # 초기 전체 WIP (보상 계산용)
        self.total_processed = 0           # 누적 처리량
        self.utilization_history = []      # 가동률 이력
        self.allocation_history = []       # 할당 이력 (출력용)
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        환경을 초기 상태로 리셋
        
        반환값:
        -------
        observation : np.ndarray
            초기 관측값
        info : Dict
            추가 정보
        """
        super().reset(seed=seed)
        
        # 초기 WIP 설정 (주어지지 않은 경우 랜덤 생성)
        if self.initial_wip is not None:
            self.current_wip = self.initial_wip.copy()
        else:
            self.current_wip = self.np_random.integers(
                10, 30, size=self.num_processes
            ).astype(np.float32)
        
        # 장비 처리 용량 설정 (각 장비가 시간 슬롯당 처리 가능한 양)
        self.resource_capacity = self.np_random.integers(
            3, 8, size=self.num_resources
        ).astype(np.float32)
        
        # 공정별 처리 시간 (1~3 시간 슬롯)
        self.process_times = self.np_random.integers(
            1, 4, size=self.num_processes
        ).astype(np.float32)
        
        # 공정별 납기일 (5~15 시간 슬롯 후)
        self.due_dates = self.np_random.integers(
            5, 16, size=self.num_processes
        ).astype(np.float32)
        
        # 상태 변수 초기화
        self.current_slot = 0
        self.total_initial_wip = float(np.sum(self.current_wip))
        self.total_processed = 0
        self.utilization_history = []
        self.allocation_history = []
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """
        현재 상태를 고정 크기 Observation으로 변환
        
        반환값:
        -------
        observation : np.ndarray
            형태: (MAX_ELEMENTS, FEATURE_DIM)
            패딩이 포함된 고정 크기 텐서
        """
        # 패딩으로 채워진 observation 초기화
        observation = np.zeros((MAX_ELEMENTS, FEATURE_DIM), dtype=np.float32)
        
        # 공정 정보 채우기 (Type_ID = 1)
        for i in range(self.num_processes):
            observation[i] = [
                1.0,                          # Type_ID: 공정
                self.current_wip[i],          # 현재 WIP 수량
                self.process_times[i],        # 처리 시간
                max(0, self.due_dates[i] - self.current_slot)  # 남은 납기일
            ]
        
        # 장비 정보 채우기 (Type_ID = 2)
        for j in range(self.num_resources):
            observation[MAX_PROCESSES + j] = [
                2.0,                          # Type_ID: 장비
                self.resource_capacity[j],    # 처리 용량
                1.0,                          # 처리 속도 (정규화됨)
                float(self.max_time_slots - self.current_slot)  # 남은 시간
            ]
        
        # 나머지는 패딩 (Type_ID = 0, 이미 0으로 초기화됨)
        return observation
    
    def _get_info(self) -> Dict:
        """
        현재 상태에 대한 추가 정보 반환
        """
        return {
            'current_slot': self.current_slot,
            'remaining_wip': np.sum(self.current_wip),
            'total_processed': self.total_processed,
            'num_processes': self.num_processes,
            'num_resources': self.num_resources
        }
    
    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """
        액션 벡터를 (공정, 장비) 할당 행렬로 디코딩
        유효하지 않은 조합은 0으로 마스킹
        
        매개변수:
        ---------
        action : np.ndarray
            형태: (MAX_PROCESSES * MAX_RESOURCES,)
            
        반환값:
        -------
        allocation : np.ndarray
            형태: (num_processes, num_resources)
            각 (공정, 장비) 조합에 할당된 수량
        """
        # 고정 크기 액션을 2D 행렬로 변환
        action_matrix = action.reshape(MAX_PROCESSES, MAX_RESOURCES)
        
        # 유효한 범위만 추출 (마스킹)
        allocation = action_matrix[:self.num_processes, :self.num_resources].copy()
        
        return allocation.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        액션을 수행하고 환경 상태를 업데이트
        
        매개변수:
        ---------
        action : np.ndarray
            MultiDiscrete 액션 (고정 크기)
            
        반환값:
        -------
        observation : np.ndarray
            다음 상태 관측값
        reward : float
            보상
        terminated : bool
            에피소드 종료 여부 (WIP 완료)
        truncated : bool
            에피소드 중단 여부 (시간 초과)
        info : Dict
            추가 정보
        """
        # 액션 디코딩
        allocation = self._decode_action(action)
        
        # 할당 이력 기록
        slot_allocation = {'slot': self.current_slot, 'allocation': allocation.copy()}
        self.allocation_history.append(slot_allocation)
        
        # 처리량 계산 및 WIP 업데이트
        slot_processed = 0
        total_capacity_used = 0
        total_capacity_available = np.sum(self.resource_capacity)
        
        for j in range(self.num_resources):
            # 장비 j의 사용량 계산
            resource_usage = 0
            
            for i in range(self.num_processes):
                # 할당량이 현재 WIP와 장비 용량을 초과하지 않도록 조정
                actual_allocation = min(
                    allocation[i, j],
                    self.current_wip[i],
                    self.resource_capacity[j] - resource_usage
                )
                
                if actual_allocation > 0:
                    self.current_wip[i] -= actual_allocation
                    slot_processed += actual_allocation
                    resource_usage += actual_allocation
            
            total_capacity_used += resource_usage
        
        # 가동률 계산 및 기록
        slot_utilization = total_capacity_used / max(total_capacity_available, 1)
        self.utilization_history.append(slot_utilization)
        self.total_processed += slot_processed
        
        # 시간 슬롯 증가
        self.current_slot += 1
        
        # 종료 조건 확인
        wip_completed = np.sum(self.current_wip) <= 0
        time_exceeded = self.current_slot >= self.max_time_slots
        
        terminated = wip_completed
        truncated = time_exceeded and not wip_completed
        
        # 보상 계산
        reward = self._calculate_reward(
            slot_processed, 
            wip_completed, 
            time_exceeded
        )
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _calculate_reward(
        self, 
        slot_processed: float, 
        wip_completed: bool, 
        time_exceeded: bool
    ) -> float:
        """
        보상 계산
        
        보상 구조:
        - Step Reward: (현재 슬롯 처리량 / 전체 목표 WIP)
        - Final Reward (성공): 총 가동률 * 10
        - Final Reward (실패): -10 패널티
        """
        # Step Reward: 처리량 비율
        step_reward = slot_processed / max(self.total_initial_wip, 1)
        
        # Final Reward
        if wip_completed:
            # 성공: 평균 가동률 기반 보너스
            avg_utilization = np.mean(self.utilization_history) if self.utilization_history else 0
            final_reward = avg_utilization * 10.0
            return step_reward + final_reward
        elif time_exceeded:
            # 실패: 패널티
            return step_reward - 10.0
        
        return step_reward
    
    def render(self):
        """환경 상태 렌더링 (콘솔 출력)"""
        if self.render_mode == 'human':
            print(f"\n=== 시간 슬롯 {self.current_slot} ===")
            print(f"현재 WIP: {self.current_wip}")
            print(f"장비 용량: {self.resource_capacity}")
            print(f"누적 처리량: {self.total_processed:.1f}")
            if self.utilization_history:
                print(f"현재 가동률: {self.utilization_history[-1]:.2%}")
    
    def get_valid_action_mask(self) -> np.ndarray:
        """
        유효한 액션을 나타내는 마스크 반환
        유효하지 않은 (공정, 장비) 조합은 False
        
        반환값:
        -------
        mask : np.ndarray
            형태: (MAX_PROCESSES * MAX_RESOURCES,)
        """
        mask = np.zeros(MAX_PROCESSES * MAX_RESOURCES, dtype=bool)
        
        for i in range(self.num_processes):
            for j in range(self.num_resources):
                idx = i * MAX_RESOURCES + j
                # 해당 공정에 WIP가 남아있는 경우에만 유효
                mask[idx] = self.current_wip[i] > 0
        
        return mask


# ==============================================================================
# Transformer Feature Extractor
# ==============================================================================
class PositionalEncoding(nn.Module):
    """
    Positional Encoding 모듈
    
    Transformer는 위치 정보를 자체적으로 학습하지 않으므로,
    입력에 위치 정보를 추가하여 토큰의 순서를 인코딩합니다.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        매개변수:
        ---------
        d_model : int
            모델의 임베딩 차원
        max_len : int
            최대 시퀀스 길이
        dropout : float
            드롭아웃 비율
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 계산 (sin/cos 방식)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # 학습되지 않는 버퍼로 등록
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력에 위치 인코딩 추가
        
        매개변수:
        ---------
        x : torch.Tensor
            형태: (batch_size, seq_len, d_model)
            
        반환값:
        -------
        x : torch.Tensor
            위치 인코딩이 추가된 텐서
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer 기반 Feature Extractor
    
    Stable-Baselines3 PPO와 통합되어 사용됩니다.
    입력 Observation을 Transformer Encoder를 통해 처리하여
    고차원 특성을 추출합니다.
    
    구조:
    1. Linear Projection: (FEATURE_DIM) -> (d_model)
    2. Positional Encoding
    3. Transformer Encoder (self-attention)
    4. Global Average Pooling
    5. Output MLP
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        매개변수:
        ---------
        observation_space : spaces.Box
            관측 공간
        features_dim : int
            출력 특성 차원
        d_model : int
            Transformer 내부 차원
        nhead : int
            멀티헤드 어텐션의 헤드 수
        num_layers : int
            Transformer Encoder 레이어 수
        dropout : float
            드롭아웃 비율
        """
        super().__init__(observation_space, features_dim)
        
        self.d_model = d_model
        
        # 1. Linear Projection: 입력 특성을 d_model 차원으로 투영
        self.input_projection = nn.Linear(FEATURE_DIM, d_model)
        
        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, MAX_ELEMENTS, dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # (batch, seq, feature) 형식 사용
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 4. Output MLP: Transformer 출력을 features_dim으로 변환
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        매개변수:
        ---------
        observations : torch.Tensor
            형태: (batch_size, MAX_ELEMENTS, FEATURE_DIM)
            
        반환값:
        -------
        features : torch.Tensor
            형태: (batch_size, features_dim)
        """
        # 배치 크기 확인
        batch_size = observations.shape[0]
        
        # Padding Mask 생성: Type_ID가 0인 토큰은 패딩
        # True = 마스킹(무시), False = 유효
        padding_mask = observations[:, :, 0] == 0  # (batch_size, MAX_ELEMENTS)
        
        # 1. Linear Projection
        x = self.input_projection(observations)  # (batch, seq, d_model)
        
        # 2. Positional Encoding
        x = self.positional_encoding(x)
        
        # 3. Transformer Encoder (with padding mask)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 4. Global Average Pooling (패딩 제외)
        # 유효한 토큰만 평균
        valid_mask = ~padding_mask  # True = 유효
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        
        x_masked = x * valid_mask_expanded
        num_valid = valid_mask_expanded.sum(dim=1).clamp(min=1)  # 0으로 나누기 방지
        x_pooled = x_masked.sum(dim=1) / num_valid  # (batch, d_model)
        
        # 5. Output MLP
        features = self.output_mlp(x_pooled)  # (batch, features_dim)
        
        return features


# ==============================================================================
# Custom Actor-Critic Policy
# ==============================================================================
class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    커스텀 Actor-Critic 정책
    
    Transformer Feature Extractor를 사용하는 PPO 정책입니다.
    SB3의 기본 MLP 대신 Transformer를 특성 추출기로 사용합니다.
    """
    
    def __init__(self, *args, **kwargs):
        """정책 초기화 (Transformer Extractor 지정)"""
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs={
                'features_dim': 128,
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'dropout': 0.1
            }
        )


# ==============================================================================
# Expert 데이터 생성 (Behavior Cloning용)
# ==============================================================================
def generate_expert_data(
    env: ManufacturingSchedulingEnv,
    num_episodes: int = 50,
    diverse_envs: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expert 규칙 기반 데이터 생성
    
    규칙: 잔여 WIP가 많은 공정에 우선적으로 장비를 배분
    
    매개변수:
    ---------
    env : ManufacturingSchedulingEnv
        학습 환경
    num_episodes : int
        생성할 에피소드 수
    diverse_envs : bool
        다양한 환경 크기에서 데이터 수집 여부 (일반화 향상)
        
    반환값:
    -------
    observations : np.ndarray
        관측값 배열
    actions : np.ndarray
        Expert 액션 배열
    """
    observations = []
    actions = []
    
    for ep_idx in range(num_episodes):
        # 다양한 환경 크기로 학습 (일반화 향상)
        if diverse_envs and ep_idx % 3 == 0:
            # 다양한 공정 수로 임시 환경 생성
            temp_num_processes = np.random.randint(2, MAX_PROCESSES + 1)
            temp_num_resources = env.num_resources
            temp_wip = np.random.randint(10, 30, size=temp_num_processes).tolist()
            temp_env = ManufacturingSchedulingEnv(
                num_processes=temp_num_processes,
                num_resources=temp_num_resources,
                initial_wip=temp_wip,
                max_time_slots=env.max_time_slots
            )
            current_env = temp_env
        else:
            current_env = env
            
        obs, _ = current_env.reset()
        done = False
        
        while not done:
            # Expert 정책: 잔여 WIP 기반 우선순위 할당
            action = expert_policy(current_env)
            
            observations.append(obs.flatten())
            actions.append(action)
            
            obs, _, terminated, truncated, _ = current_env.step(action)
            done = terminated or truncated
    
    return np.array(observations), np.array(actions)


def expert_policy(env: ManufacturingSchedulingEnv) -> np.ndarray:
    """
    Expert 정책: 잔여 WIP가 많은 공정에 우선 배분
    
    규칙:
    1. 공정을 잔여 WIP 순으로 정렬 (내림차순)
    2. 각 장비의 용량을 WIP가 많은 공정부터 할당
    3. 장비 용량이 소진될 때까지 반복
    
    매개변수:
    ---------
    env : ManufacturingSchedulingEnv
        현재 환경 상태
        
    반환값:
    -------
    action : np.ndarray
        Expert 액션 (고정 크기)
    """
    # 고정 크기 액션 초기화
    action = np.zeros(MAX_PROCESSES * MAX_RESOURCES, dtype=np.int64)
    
    # 공정을 WIP 순으로 정렬 (많은 순)
    process_priority = np.argsort(-env.current_wip)
    
    # 각 장비의 남은 용량
    remaining_capacity = env.resource_capacity.copy()
    
    # 우선순위 순으로 공정에 장비 할당
    for proc_idx in process_priority:
        if proc_idx >= env.num_processes:
            continue
            
        remaining_wip = env.current_wip[proc_idx]
        
        for res_idx in range(env.num_resources):
            if remaining_wip <= 0 or remaining_capacity[res_idx] <= 0:
                continue
            
            # 할당 가능한 최대량
            allocation = min(
                remaining_wip,
                remaining_capacity[res_idx],
                MAX_COUNT - 1  # MAX_COUNT 제한
            )
            
            if allocation > 0:
                # 액션 인덱스 계산 (고정 크기 기준)
                action_idx = proc_idx * MAX_RESOURCES + res_idx
                action[action_idx] = int(allocation)
                
                remaining_wip -= allocation
                remaining_capacity[res_idx] -= allocation
    
    return action


# ==============================================================================
# Behavior Cloning 사전 학습
# ==============================================================================
def behavior_cloning_pretrain(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3
) -> List[float]:
    """
    Behavior Cloning으로 정책 네트워크 사전 학습
    
    Expert 데이터를 사용하여 Supervised Learning으로
    정책 네트워크를 초기화합니다.
    
    매개변수:
    ---------
    model : PPO
        사전 학습할 PPO 모델
    observations : np.ndarray
        Expert 관측값
    actions : np.ndarray
        Expert 액션
    epochs : int
        학습 에폭 수
    batch_size : int
        배치 크기
    learning_rate : float
        학습률
        
    반환값:
    -------
    losses : List[float]
        에폭별 손실 값
    """
    print("\n" + "="*60)
    print("Behavior Cloning 사전 학습 시작")
    print("="*60)
    
    # 데이터 준비
    device = model.device
    obs_tensor = torch.FloatTensor(observations).to(device)
    
    # Observation을 원래 형태로 복원
    obs_tensor = obs_tensor.view(-1, MAX_ELEMENTS, FEATURE_DIM)
    
    # 액션 텐서 (MultiDiscrete이므로 각 차원별 처리)
    action_tensor = torch.LongTensor(actions).to(device)
    
    # DataLoader 생성
    dataset = TensorDataset(obs_tensor, action_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer (정책 네트워크만)
    optimizer = optim.Adam(
        model.policy.parameters(),
        lr=learning_rate
    )
    
    # 손실 함수: Cross Entropy (각 액션 차원별)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_obs, batch_actions in dataloader:
            optimizer.zero_grad()
            
            # 정책 네트워크의 action distribution 획득
            # SB3 정책의 내부 구조 활용
            features = model.policy.extract_features(batch_obs)
            if hasattr(model.policy, 'mlp_extractor'):
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                latent_pi = features
            
            # Action distribution의 logits 획득
            action_logits = model.policy.action_net(latent_pi)
            
            # MultiDiscrete 액션 처리
            # action_logits: (batch, MAX_PROCESSES * MAX_RESOURCES * MAX_COUNT)
            # 각 (공정, 장비) 조합마다 MAX_COUNT개의 logit이 있음
            
            total_loss = 0.0
            num_dims = MAX_PROCESSES * MAX_RESOURCES
            
            for dim in range(num_dims):
                # 해당 차원의 logits 추출
                start_idx = dim * MAX_COUNT
                end_idx = (dim + 1) * MAX_COUNT
                dim_logits = action_logits[:, start_idx:end_idx]
                
                # 해당 차원의 타겟 액션
                dim_targets = batch_actions[:, dim]
                
                # Cross Entropy 손실
                dim_loss = criterion(dim_logits, dim_targets)
                total_loss += dim_loss
            
            # 평균 손실
            loss = total_loss / num_dims
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        # 진행 상황 출력 (10 에폭마다)
        if (epoch + 1) % 10 == 0:
            print(f"  에폭 {epoch + 1}/{epochs} - 손실: {avg_loss:.4f}")
    
    print("Behavior Cloning 사전 학습 완료!")
    print("="*60)
    
    return losses


# ==============================================================================
# 가변 환경 래퍼 (일반화를 위한 다양한 공정 수 사용)
# ==============================================================================
class VariableSizeEnvWrapper(gym.Wrapper):
    """
    가변 크기 환경 래퍼
    
    에피소드마다 다른 공정 수를 사용하여 일반화 성능을 향상시킵니다.
    이를 통해 학습된 정책이 다양한 크기의 환경에서도 작동할 수 있습니다.
    """
    
    def __init__(
        self,
        env: ManufacturingSchedulingEnv,
        min_processes: int = 2,
        max_processes: int = 4
    ):
        """
        매개변수:
        ---------
        env : ManufacturingSchedulingEnv
            기본 환경
        min_processes : int
            최소 공정 수
        max_processes : int
            최대 공정 수
        """
        super().__init__(env)
        self.min_processes = min_processes
        self.max_processes = max_processes
        self.base_num_resources = env.num_resources
        self.base_max_time_slots = env.max_time_slots
        
    def reset(self, seed=None, options=None):
        """에피소드마다 랜덤한 공정 수 사용"""
        # 랜덤한 공정 수 선택
        if seed is not None:
            np.random.seed(seed)
        
        new_num_processes = np.random.randint(self.min_processes, self.max_processes + 1)
        
        # 환경 파라미터 업데이트
        self.env.num_processes = new_num_processes
        
        # 초기 WIP 재생성
        self.env.initial_wip = np.random.randint(
            10, 30, size=new_num_processes
        ).astype(np.float32)
        
        return self.env.reset(seed=seed, options=options)


# ==============================================================================
# Tensorboard 콜백
# ==============================================================================
class TensorboardCallback(BaseCallback):
    """
    학습 과정을 Tensorboard로 로깅하는 콜백
    
    로깅 항목:
    - 에피소드 보상
    - 에피소드 길이
    - 가동률
    - 처리량
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """매 스텝마다 호출"""
        # 에피소드 종료 시 로깅
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                self.logger.record('custom/episode_reward', ep_info['r'])
            if 'l' in ep_info:
                self.logger.record('custom/episode_length', ep_info['l'])
        
        return True


# ==============================================================================
# 메인 실행 함수
# ==============================================================================
def print_allocation_results(
    allocation_history: List[Dict],
    num_processes: int,
    num_resources: int
):
    """
    할당 결과를 사람이 읽기 쉬운 형태로 출력
    
    매개변수:
    ---------
    allocation_history : List[Dict]
        시간 슬롯별 할당 이력
    num_processes : int
        공정 수
    num_resources : int
        장비 수
    """
    print("\n" + "="*70)
    print("시간 슬롯별 장비 할당 결과")
    print("="*70)
    
    # 헤더 출력
    header = "슬롯 |"
    for j in range(num_resources):
        header += f" 장비{j+1} |"
    print(header)
    print("-" * len(header))
    
    for record in allocation_history:
        slot = record['slot']
        allocation = record['allocation']
        
        row = f" {slot:2d}  |"
        for j in range(num_resources):
            # 해당 장비에 할당된 공정들
            assignments = []
            for i in range(num_processes):
                if allocation[i, j] > 0:
                    assignments.append(f"P{i+1}:{int(allocation[i, j])}")
            
            cell = ", ".join(assignments) if assignments else "-"
            row += f" {cell:^6} |"
        
        print(row)
    
    print("="*70)
    print("(P1:3 = 공정1에 3개 할당)")


def run_inference(
    model: PPO,
    env: ManufacturingSchedulingEnv,
    render: bool = True
) -> Tuple[float, float, List[Dict]]:
    """
    학습된 모델로 추론 수행
    
    매개변수:
    ---------
    model : PPO
        학습된 PPO 모델
    env : ManufacturingSchedulingEnv
        추론 환경
    render : bool
        렌더링 여부
        
    반환값:
    -------
    total_reward : float
        총 보상
    avg_utilization : float
        평균 가동률
    allocation_history : List[Dict]
        할당 이력
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        # 모델 추론
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        done = terminated or truncated
        
        if render:
            env.render()
    
    # 결과 계산
    avg_utilization = np.mean(env.utilization_history) if env.utilization_history else 0
    
    return total_reward, avg_utilization, env.allocation_history


def main():
    """
    메인 실행 함수
    
    단계:
    1. 환경 생성 (공정 2개, 장비 2개)
    2. Expert 데이터 생성 및 Behavior Cloning 사전 학습
    3. PPO 학습 (5,000 Steps)
    4. 학습 환경에서 추론
    5. 확장 환경 (공정 3개, 장비 2개)에서 추론
    """
    print("\n" + "="*70)
    print("제조 스케줄링 RL/Transformer 시스템")
    print("="*70)
    
    # =========================================================================
    # 1단계: 학습 환경 생성 (공정 2개, 장비 2개)
    # =========================================================================
    print("\n[1단계] 학습 환경 생성 (공정 2개, 장비 2개)")
    
    train_env = ManufacturingSchedulingEnv(
        num_processes=2,
        num_resources=2,
        initial_wip=None,  # 랜덤 WIP로 다양성 확보
        max_time_slots=20,
        render_mode=None
    )
    
    # Expert 데이터 생성용 기본 환경
    base_train_env = ManufacturingSchedulingEnv(
        num_processes=2,
        num_resources=2,
        initial_wip=None,
        max_time_slots=20,
        render_mode=None
    )
    
    print(f"  - Observation Space: {train_env.observation_space.shape}")
    print(f"  - Action Space: {train_env.action_space}")
    
    # =========================================================================
    # 2단계: Expert 데이터 생성 (다양한 환경 크기 포함)
    # =========================================================================
    print("\n[2단계] Expert 데이터 생성 (100 에피소드, 다양한 환경 크기 포함)")
    
    expert_obs, expert_actions = generate_expert_data(
        base_train_env,   # 기본 환경 사용 (내부에서 다양한 크기 생성)
        num_episodes=100,
        diverse_envs=True  # 다양한 공정 수로 일반화 향상
    )
    print(f"  - 생성된 데이터: {len(expert_obs)} 샘플")
    
    # =========================================================================
    # 3단계: PPO 모델 생성
    # =========================================================================
    print("\n[3단계] PPO 모델 생성 (Transformer Feature Extractor)")
    
    model = PPO(
        policy=CustomActorCriticPolicy,
        env=train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        seed=RANDOM_SEED
    )
    
    print("  - 모델 구조:")
    print(f"    * Feature Extractor: TransformerEncoder (d_model=64, nhead=4, layers=2)")
    print(f"    * Policy: MLP (128 -> 64 -> action_dim)")
    
    # =========================================================================
    # 4단계: Behavior Cloning 사전 학습 (100 Epoch)
    # =========================================================================
    print("\n[4단계] Behavior Cloning 사전 학습 (100 Epoch)")
    
    bc_losses = behavior_cloning_pretrain(
        model=model,
        observations=expert_obs,
        actions=expert_actions,
        epochs=100,
        batch_size=64,
        learning_rate=1e-3
    )
    
    print(f"  - 최종 손실: {bc_losses[-1]:.4f}")
    
    # =========================================================================
    # 5단계: PPO 학습 (5,000 Steps)
    # =========================================================================
    print("\n[5단계] PPO 학습 (5,000 Steps)")
    print("  - Tensorboard 로깅: ./tensorboard_logs/")
    
    callback = TensorboardCallback()
    
    model.learn(
        total_timesteps=5000,
        callback=callback,
        progress_bar=False
    )
    
    print("  - PPO 학습 완료!")
    
    # =========================================================================
    # 6단계: 학습 환경에서 추론
    # =========================================================================
    print("\n[6단계] 학습 환경에서 추론 (공정 2개, 장비 2개)")
    
    train_env_test = ManufacturingSchedulingEnv(
        num_processes=2,
        num_resources=2,
        initial_wip=[20, 15],
        max_time_slots=20,
        render_mode=None
    )
    
    total_reward, avg_util, allocation_history = run_inference(
        model, train_env_test, render=False
    )
    
    print(f"\n  결과:")
    print(f"  - 총 보상: {total_reward:.2f}")
    print(f"  - 평균 가동률: {avg_util:.2%}")
    print(f"  - 총 처리 슬롯: {len(allocation_history)}")
    
    # 할당 결과 출력
    print_allocation_results(
        allocation_history,
        train_env_test.num_processes,
        train_env_test.num_resources
    )
    
    # =========================================================================
    # 7단계: 확장 환경에서 추론 (공정 3개, 장비 2개)
    # =========================================================================
    print("\n[7단계] 확장 환경에서 추론 (공정 3개, 장비 2개)")
    print("  - 동일한 모델을 사용하여 더 큰 환경에서 일반화 테스트")
    
    extended_env = ManufacturingSchedulingEnv(
        num_processes=3,
        num_resources=2,
        initial_wip=[20, 15, 25],  # 새로운 공정 추가
        max_time_slots=25,
        render_mode=None
    )
    
    total_reward_ext, avg_util_ext, allocation_history_ext = run_inference(
        model, extended_env, render=False
    )
    
    print(f"\n  결과:")
    print(f"  - 총 보상: {total_reward_ext:.2f}")
    print(f"  - 평균 가동률: {avg_util_ext:.2%}")
    print(f"  - 총 처리 슬롯: {len(allocation_history_ext)}")
    
    # 할당 결과 출력
    print_allocation_results(
        allocation_history_ext,
        extended_env.num_processes,
        extended_env.num_resources
    )
    
    # =========================================================================
    # 최종 요약
    # =========================================================================
    print("\n" + "="*70)
    print("최종 요약")
    print("="*70)
    print(f"\n학습 환경 (2 공정, 2 장비):")
    print(f"  - 평균 가동률: {avg_util:.2%}")
    print(f"  - 총 보상: {total_reward:.2f}")
    print(f"  - WIP 완료: {'예' if total_reward > 0 else '아니오'}")
    
    print(f"\n확장 환경 (3 공정, 2 장비):")
    print(f"  - 평균 가동률: {avg_util_ext:.2%}")
    print(f"  - 총 보상: {total_reward_ext:.2f}")
    print(f"  - WIP 완료: {'예' if total_reward_ext > 0 else '아니오'}")
    
    print("\n" + "-"*70)
    print("일반화 테스트 분석:")
    print("-"*70)
    if total_reward_ext < 0:
        print("  * 확장 환경에서 성능이 저하된 이유:")
        print("    - PPO는 학습 환경(공정 2개)에서만 학습됨")
        print("    - 공정 3(P3)에 대한 액션 정책이 학습되지 않음")
        print("    - 이는 RL의 일반적인 한계이며, 다양한 환경에서")
        print("      학습하거나 추가 학습이 필요함")
        print("\n  * 개선 방안:")
        print("    - 학습 시 다양한 공정 수 사용 (Curriculum Learning)")
        print("    - 더 많은 Expert 데이터로 BC 사전학습 강화")
        print("    - 확장 환경에서 Fine-tuning 수행")
    else:
        print("  * 확장 환경에서도 양호한 일반화 성능을 보임")
    
    print("\n" + "="*70)
    print("구현 특징:")
    print("="*70)
    print("  1. Transformer Feature Extractor: d_model=64, nhead=4, layers=2")
    print("  2. Positional Encoding: Sinusoidal 방식")
    print("  3. Padding Mask: 가변 길이 시퀀스 처리 지원")
    print("  4. Behavior Cloning: 100 Epoch 사전학습")
    print("  5. PPO: 5,000 Steps 강화학습")
    print("  6. 고정 크기 Observation/Action: 가변 환경 일반화 지원")
    
    print("\n" + "="*70)
    print("실행 완료!")
    print("="*70)
    
    return model


if __name__ == "__main__":
    # 실행
    trained_model = main()
