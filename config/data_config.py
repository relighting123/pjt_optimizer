import pandas as pd

# 1. 제품별 일일 계획량 (Daily Demand)
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

# 4. 처리 소요 시간 (Processing Time - Unit: Minutes)
# 단위당 처리 시간을 분 단위로 설정
PROCESS_CONFIG = {
    ('Product_A', 'OP10', 'Model_X'): 1.5, # 1.5분
    ('Product_B', 'OP10', 'Model_X'): 2.0,
    ('Product_A', 'OP20', 'Model_Y'): 2.5,
    ('Product_B', 'OP20', 'Model_Y'): 3.0,
}

# 5. 장비 가용 시간 (Available Time per Day - Unit: Minutes)
# 24시간 = 1440분
AVAILABLE_TIME = 1440

# 6. 재공량 (Input-based WIP)
WIP = {
    ('Product_A', 'OP10'): 200,
    ('Product_A', 'OP20'): 0,
    ('Product_B', 'OP10'): 200,
    ('Product_B', 'OP20'): 0,
}

# 7. 장비별 현재 작업 중인 정보 (Equipment WIP - Unit: Minutes)
# End_Time_Offset: 현재 시각으로부터 몇 분 뒤에 끝나는지
EQP_WIP = {
    'Unit_1': {'Product': 'Product_A', 'Operation': 'OP10', 'End_Time_Offset': 10}, 
    'Unit_3': {'Product': 'Product_B', 'Operation': 'OP20', 'End_Time_Offset': 5},
}

# 8. 툴 정보 (Tool Constraints)
# (Product, Operation): Available Count
# 장비 전환 시 툴은 반환되며, 동시 사용 대수만 제한함
TOOLS = {
    ('Product_A', 'OP10'): 1, 
    ('Product_B', 'OP10'): 2,
    ('Product_A', 'OP20'): 2,
    ('Product_B', 'OP20'): 2,
}

# 9. 전환 시간 설정 (Changeover Times - Unit: Minutes)
CHANGEOVER_CONFIG = {
    'PRODUCT_SWITCH': 30, # 30분
    'OPER_SWITCH': 30,
    'EXCEPTIONS': {}
}
