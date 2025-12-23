import pandas as pd

# 1. 제품별 계획량 (Demand)
DEMAND = {
    'Product_A': 100,
    'Product_B': 100
}

# 2. 공정 구성 (Operations)
OPERATIONS = ['OP10', 'OP20']

# 3. 장비 정보 (Equipment)
EQUIPMENT_MODELS = {
    'Model_X': ['Unit_1', 'Unit_2'],
    'Model_Y': ['Unit_3', 'Unit_4']
}

# 4. 처리 소요 시간 (Processing Time)
PROCESS_CONFIG = {
    ('Product_A', 'OP10', 'Model_X'): 100,
    ('Product_B', 'OP10', 'Model_X'): 100,
    ('Product_A', 'OP20', 'Model_Y'): 100,
    ('Product_B', 'OP20', 'Model_Y'): 100,
}

# 5. 장비 가용 시간 (Total Seconds per Shift)
AVAILABLE_TIME = 11000

# 6. 재공량 (Input-based WIP)
WIP = {
    ('Product_A', 'OP10'): 100,
    ('Product_A', 'OP20'): 0,
    ('Product_B', 'OP10'): 100,
    ('Product_B', 'OP20'): 0,
}

# 7. 장비별 현재 작업 중인 정보 (Equipment WIP)
# EQP_ID: {Product, Operation, End_Time} - 현재 시각 기준 남은 시간 계산용
# 예: Unit_1은 'Product_A'의 'OP10'을 작업 중이며, 1000초 뒤에 끝남 (상대 시간으로 변환해서 관리)
EQP_WIP = {
    'Unit_1': {'Product': 'Product_A', 'Operation': 'OP10', 'End_Time_Offset': 500}, 
    'Unit_3': {'Product': 'Product_B', 'Operation': 'OP20', 'End_Time_Offset': 0},
}

# 8. 툴 정보 (Tool Constraints)
# (Product, Operation): Available Count
# 같은 (제품, 공정)을 동시에 진행할 수 있는 장비 대수를 제한
TOOLS = {
    ('Product_A', 'OP10'): 1, # Product A의 OP10용 툴은 1개뿐이므로 한 대만 가동 가능
    ('Product_B', 'OP10'): 2,
    ('Product_A', 'OP20'): 2,
    ('Product_B', 'OP20'): 2,
}

# 9. 전환 시간 설정 (Changeover Times)
CHANGEOVER_CONFIG = {
    'PRODUCT_SWITCH': 2000,
    'OPER_SWITCH': 2000,
    'EXCEPTIONS': {}
}
