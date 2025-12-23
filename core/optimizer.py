from pulp import *
import pandas as pd
from datetime import datetime, timedelta
from config import data_config

def get_changeover_time(p_old, o_old, p_new, o_new):
    """
    제품/공정 변경에 따른 전환 시간 계산
    """
    if p_old is None: return 0
    
    # 1. 예외 케이스 체크
    if (p_old, p_new, o_new) in data_config.CHANGEOVER_CONFIG['EXCEPTIONS']:
        return data_config.CHANGEOVER_CONFIG['EXCEPTIONS'][(p_old, p_new, o_new)]
    
    # 2. 제품 변경 시
    if p_old != p_new:
        return data_config.CHANGEOVER_CONFIG['PRODUCT_SWITCH']
    
    # 3. 제품은 같으나 공정만 변경 시
    if o_old != o_new:
        return data_config.CHANGEOVER_CONFIG['OPER_SWITCH']
        
    return 0

def solve_production_allocation(demands=None, eqp_models=None, proc_config=None, avail_time=None, opers_list=None, wip=None, eqp_wip=None, tools=None):
    # 인자가 제공되지 않으면 data_config의 기본값 사용
    demands = demands or data_config.DEMAND
    eqp_models = eqp_models or data_config.EQUIPMENT_MODELS
    proc_config = proc_config or data_config.PROCESS_CONFIG
    avail_time = avail_time or data_config.AVAILABLE_TIME
    opers_list = opers_list or data_config.OPERATIONS
    wip = wip or data_config.WIP
    eqp_wip = eqp_wip or {}
    tools = tools or {}
    
    # 1. 문제 정의
    prob = LpProblem("Production_Line_Balancing", LpMinimize)

    # 결정 변수 정의
    units = [unit for model in eqp_models for unit in eqp_models[model]]
    
    # 가능한 (Prod, Oper, Unit) 조합 추출
    valid_combinations = []
    for (p, o, m), t in proc_config.items():
        if m in eqp_models:
            for u in eqp_models[m]:
                valid_combinations.append((p, o, u))

    qty_vars = LpVariable.dicts("Qty", valid_combinations, lowBound=0, cat='Continuous')
    assign_vars = LpVariable.dicts("Assign", valid_combinations, cat='Binary')
    unmet_vars = LpVariable.dicts("Unmet", [(p, o) for p in demands for o in opers_list], lowBound=0, cat='Continuous')
    
    # 3. 목적 함수 설계
    # 1순위: 미충족 수요 최소화 (Penalty 1,000,000)
    # 2순위: 현재 작업 중인 장비의 작업 연속성 유지 (Penalty 10,000)
    # 3순위: 제품/공정 전환 최소화 (할당 개수 1개당 Penalty 1,000)
    # 4순위: 불필요한 과잉 생산 최소화 (수량 1개당 Penalty 1)
    p_unmet = 1000000
    p_continuation = 10000  # 작업 연속성 페널티 (새로 추가)
    p_assign = 1000
    p_qty = 1
    
    # 작업 연속성 페널티 계산
    # 현재 작업 중인 장비가 다른 (제품, 공정)을 할당받으면 페널티 부과
    continuation_penalty = 0
    for u in units:
        if u in eqp_wip:
            current_prod = eqp_wip[u]['Product']
            current_oper = eqp_wip[u]['Operation']
            
            # 이 장비에 할당 가능한 모든 조합 중, 현재 작업과 다른 것에 페널티
            for (p, o, unit) in valid_combinations:
                if unit == u and (p != current_prod or o != current_oper):
                    continuation_penalty += p_continuation * assign_vars[p, o, unit]
    
    prob += (p_unmet * lpSum([unmet_vars[p, o] for p, o in unmet_vars]) + 
             continuation_penalty +
             p_assign * lpSum([assign_vars[p, o, u] for (p, o, u) in valid_combinations]) +
             p_qty * lpSum([qty_vars[p, o, u] for (p, o, u) in valid_combinations]))

    # 4. 제약 조건 설정
    for (p, o, u) in valid_combinations:
        prob += qty_vars[p, o, u] <= 100000 * assign_vars[p, o, u]
    
    # B. 수요 충족 제약 (마지막 공정 생산량 + 마지막 공정 재공량 + 미충족량 >= 수요)
    last_oper = opers_list[-1]
    for p, demand in demands.items():
        relevant_units = [u for (prod, oper, u) in qty_vars if prod == p and oper == last_oper]
        wip_val = wip.get((p, last_oper), 0)
        prob += lpSum([qty_vars[p, last_oper, u] for u in relevant_units]) + wip_val + unmet_vars[p, last_oper] >= demand

    # C. 공정 수순 흐름 제약 (Flow Conservation + WIP)
    # 각 공정의 생산량은 (해당 공정 시작 전 대기 재공 + 전 공정 생산량)을 초과할 수 없음
    for p in demands:
        for i, curr_op in enumerate(opers_list):
            curr_units_p = [u for (prod, oper, u) in qty_vars if prod == p and oper == curr_op]
            wip_val = wip.get((p, curr_op), 0) # 현재 공정을 진행하기 위해 대기 중인 재공
            
            if i == 0:
                # 첫 공정: 투입 가능한 재공(원소재 등)만큼만 생산 가능
                prob += lpSum([qty_vars[p, curr_op, u] for u in curr_units_p]) <= wip_val
            else:
                # 이후 공정: (해당 공정 대기 재공 + 전 공정에서 넘어온 생산량)만큼 생산 가능
                prev_op = opers_list[i-1]
                prev_units_p = [u for (prod, oper, u) in qty_vars if prod == p and oper == prev_op]
                
                prob += (lpSum([qty_vars[p, curr_op, u] for u in curr_units_p]) <= 
                         wip_val + lpSum([qty_vars[p, prev_op, u] for u in prev_units_p]))

    # D. 툴 제약 (Tool-Hour Capacity)
    # 특정 (제품, 공정) 작업의 총 소요 시간(장비들이 나눠서 하는 시간의 합)은 가용한 툴-시간(Tool-Hours)을 초과할 수 없음
    # 이는 툴 교체 후 반환하여 다른 장비가 순차적으로 사용하는 '반환 및 재사용' 논리를 반영함
    for (p, o) in [(prod, oper) for prod in demands for oper in opers_list]:
        tool_qty = tools.get((p, o), 99) # 기본값 99 (제한 없음)
        
        # 해당 (p, o)를 수행할 수 있는 모든 (장비, 소요시간) 조합 찾기
        prod_tasks = []
        for (prod, oper, model), t in proc_config.items():
            if prod == p and oper == o:
                # 이 (p, o)를 처리할 수 있는 모든 유닛들의 수량 변수 확인
                for u in units:
                    if model in eqp_models and u in eqp_models[model]:
                        prod_tasks.append((u, t))
        
        if prod_tasks:
            # 총 소요 시간 (분) = Sum(수량 * 사이클타임)
            total_tool_time_needed = lpSum([qty_vars[p, o, u] * t for (u, t) in prod_tasks])
            # 가용 툴-시간 (분) = 툴 개수 * 장비 가용 시간(1440분)
            prob += total_tool_time_needed <= tool_qty * avail_time

    # E. 장비 가용 시간 제약 (Capacity + EQP WIP)
    for u in units:
        # 해당 장비의 현재 작업 종료 시각(End_Time_Offset - 분 단위)을 고려하여 가용 시간 차감
        occupied_min = 0
        if u in eqp_wip:
            occupied_min = eqp_wip[u].get('End_Time_Offset', 0)
        
        effective_avail_time = avail_time - occupied_min
        
        assigned_tasks = []
        for (p, o, m), t in proc_config.items():
            if m in eqp_models and u in eqp_models[m]:
                assigned_tasks.append((p, o, t))
        
        if assigned_tasks:
            total_unit_time = lpSum([qty_vars[p, o, u] * t for (p, o, t) in assigned_tasks])
            # (이번에 할당된 작업 시간) <= (실제 사용 가능한 남은 분)
            prob += total_unit_time <= effective_avail_time

    # 5. 최적화 실행
    status = prob.solve(PULP_CBC_CMD(msg=0))
    print(f"[Debug] Solver Status: {LpStatus[status]}")

    # 6. 결과 정리
    if LpStatus[status] in ['Optimal', 'Not Solved']:
        results = []
        now = datetime.now()
        
        # 장비별 마지막 상태 초기화 (EQP WIP 반영)
        unit_last_state = {}
        for u in units:
            if u in eqp_wip:
                info = eqp_wip[u]
                unit_last_state[u] = {
                    'prod': info['Product'], 
                    'oper': info['Operation'], 
                    'time': now + timedelta(minutes=info['End_Time_Offset'])
                }
            else:
                unit_last_state[u] = {'prod': None, 'oper': None, 'time': now}
        
        for (p, o, u) in sorted(valid_combinations, key=lambda x: (x[2], x[0])):
            q = value(qty_vars[p, o, u])
            if q > 1e-5:
                model_name = next(m for m, u_list in eqp_models.items() if u in u_list)
                unit_time = proc_config[(p, o, model_name)]
                
                last_info = unit_last_state[u]
                co_min = get_changeover_time(last_info['prod'], last_info['oper'], p, o)
                
                if co_min > 0:
                    co_start = last_info['time']
                    co_end = co_start + timedelta(minutes=co_min)
                    results.append({
                        'Unit': u, 'Product': 'CHANGEOVER', 'Operation': 'SETUP',
                        'Quantity': 0, 'Time_Spent_Min': co_min,
                        'Start_Time': co_start, 'End_Time': co_end, 'Type': 'Setup'
                    })
                    last_info['time'] = co_end
                
                prod_start = last_info['time']
                spent_time_min = q * unit_time
                prod_end = prod_start + timedelta(minutes=spent_time_min)
                
                results.append({
                    'Unit': u, 'Product': p, 'Operation': o,
                    'Quantity': q, 'Time_Spent_Min': spent_time_min,
                    'Start_Time': prod_start, 'End_Time': prod_end, 'Type': 'Production'
                })
                unit_last_state[u] = {'prod': p, 'oper': o, 'time': prod_end}
        
        unmet_results = []
        for (p, o), var in unmet_vars.items():
            val = value(var)
            if val > 1e-5:
                unmet_results.append({'Product': p, 'Operation': o, 'Unmet_Qty': val})
        
        df_res = pd.DataFrame(results)
        max_workload = 0
        if not df_res.empty:
            max_workload = df_res.groupby('Unit')['Time_Spent_Min'].sum().max()
        
        return df_res, max_workload, pd.DataFrame(unmet_results)
    else:
        return None, 0, pd.DataFrame()
