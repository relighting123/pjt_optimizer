# Production Allocation & Balancer

이 프로젝트는 혼합 정수 선형 계획법(MILP)을 사용하여 장비별 생산 할당을 최적화하고, 간트 차트를 통해 타임라인을 시각화하며, Oracle DB 및 API 연동을 통해 자동화를 지원하는 시스템입니다.

---

## 🚀 시작 가이드 (Setup Guide)

가상 환경을 구성하고 필수 패키지를 설치하는 방법입니다.

### 1. 가상 환경 생성 및 활성화
```powershell
# 프로젝트 폴더로 이동
cd production_balancer

# 가상 환경 생성 (Windows/macOS/Linux 공통)
python -m venv venv

# 가상 환경 활성화 (Windows)
.\venv\Scripts\activate

# 가상 환경 활성화 (macOS/Linux)
# source venv/bin/activate
```

### 2. 패키지 설치
```powershell
pip install -r requirements.txt
```

---

## 🧩 프로젝트 구조 (Folder Structure)

```text
production_balancer/
├── core/                # 핵심 비즈니스 로직
│   ├── optimizer.py     # MILP 최적화 엔진
│   └── job_manager.py   # 비동기 큐 및 워커 관리
├── database/            # 데이터베이스 레이어
│   └── manager.py       # Oracle DB 연동 (I/O)
├── config/              # 설정 및 데이터 정의
│   ├── data_config.py   # 하드코딩된 기초 데이터 (Sample)
│   └── config.yaml      # DB 접속 및 시스템 설정
├── app.py               # Streamlit 대시보드 (시각화/검증)
├── api.py               # FastAPI 서버 (자동화/연동)
├── main.py              # CLI 실행 엔트리
└── requirements.txt     # 의존성 패키지 목록
```

---

## 🧠 모델링 방법 (Modeling Approach)

이 시스템은 **혼합 정수 선형 계획법(MILP)**을 사용하여 최적의 해를 도출합니다.

### 1. 목표 함수 (Objective Function)
1.  **미충족 수요 최소화 (Maximize Plan Achievement)**: 생산 부족분(Unmet Demand)에 매우 높은 가중치(Penalty)를 부여하여 모든 수요를 충족하는 해를 우선적으로 찾습니다.
2.  **전환 최소화 (Minimize Changeovers)**: 장비에 할당된 작업 개수(Assignments)에 페널티를 부여하여, 가급적 한 장비가 한 제품/공정을 연속해서 맡도록 유도합니다.

### 2. 주요 제약 조건 (Constraints)
-   **수요 제약**: 최종 공정의 생산량 합계가 목표 수요를 충족해야 함.
-   **수순 흐름 제약 (Flow Conservation)**: 선행 공정의 생산량이 후행 공정의 생산량보다 크거나 같아야 함.
-   **가용 시간 제약**: (생산 시간 + 전환 시간)의 합계가 장비의 운용 가능 시간(AVAILABLE_TIME)을 초과할 수 없음.
-   **장비 호환성**: 각 공정은 지정된 장비 모델에서만 수행 가능함.

### 3. 전환(Changeover) 로직
-   **제품 전환**: 장비가 다른 제품군으로 변경될 때 2시간(기본값)의 Non-productive 타임 발생.
-   **공정 전환**: 동일 제품 내에서 공정이 변경될 때 1시간(기본값) 발생.
-   **예외 케이스**: 특정 제품 간 전환 시에는 5분 이내로 처리되도록 세부 설정 가능.

---

## 💻 실행 방법 (Usage)

### 대시보드 (시각화 및 검증용)
```powershell
streamlit run app.py
```

### API 서버 (외부 연동 및 자동화용)
```powershell
python api.py
# Swagger 문서: http://localhost:8000/docs
```

### CLI 자동화 테스트
```powershell
python main.py
```
