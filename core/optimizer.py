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

def solve_production_allocation(demands=None, eqp_models=None, proc_config=None, avail_time=None, opers_list=None):
    # 인자가 제공되지 않으면 data_config의 기본값 사용
    demands = demands or data_config.DEMAND
    eqp_models = eqp_models or data_config.EQUIPMENT_MODELS
    proc_config = proc_config or data_config.PROCESS_CONFIG
    avail_time = avail_time or data_config.AVAILABLE_TIME
    opers_list = opers_list or data_config.OPERATIONS
    
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
    p_unmet = 1000000
    p_assign = 1000
    prob += (p_unmet * lpSum([unmet_vars[p, o] for p, o in unmet_vars]) + 
             p_assign * lpSum([assign_vars[p, o, u] for (p, o, u) in valid_combinations]))

    # 4. 제약 조건 설정
    for (p, o, u) in valid_combinations:
        prob += qty_vars[p, o, u] <= 100000 * assign_vars[p, o, u]
    
    last_oper = opers_list[-1]
    for p, demand in demands.items():
        relevant_units = [u for (prod, oper, u) in qty_vars if prod == p and oper == last_oper]
        prob += lpSum([qty_vars[p, last_oper, u] for u in relevant_units]) + unmet_vars[p, last_oper] >= demand

    for p in demands:
        for i in range(len(opers_list) - 1):
            curr_op = opers_list[i]
            next_op = opers_list[i+1]
            curr_units = [u for (prod, oper, u) in qty_vars if prod == p and oper == curr_op]
            next_units = [u for (prod, oper, u) in qty_vars if prod == p and oper == next_op]
            prob += lpSum([qty_vars[p, curr_op, u] for u in curr_units]) >= lpSum([qty_vars[p, next_op, u] for u in next_units])

    for u in units:
        assigned_tasks = []
        for (p, o, m), t in proc_config.items():
            if m in eqp_models and u in eqp_models[m]:
                assigned_tasks.append((p, o, t))
        if assigned_tasks:
            total_unit_time = lpSum([qty_vars[p, o, u] * t for (p, o, t) in assigned_tasks])
            prob += total_unit_time <= avail_time

    # 5. 최적화 실행
    status = prob.solve(PULP_CBC_CMD(msg=0))
    print(f"[Debug] Solver Status: {LpStatus[status]}")

    # 6. 결과 정리
    if LpStatus[status] in ['Optimal', 'Not Solved']:
        results = []
        now = datetime.now()
        unit_last_state = {u: {'prod': None, 'oper': None, 'time': now} for u in units}
        
        for (p, o, u) in sorted(valid_combinations, key=lambda x: (x[2], x[0])):
            q = value(qty_vars[p, o, u])
            if q > 1e-5:
                model_name = next(m for m, u_list in eqp_models.items() if u in u_list)
                unit_time = proc_config[(p, o, model_name)]
                
                last_info = unit_last_state[u]
                co_sec = get_changeover_time(last_info['prod'], last_info['oper'], p, o)
                
                if co_sec > 0:
                    co_start = last_info['time']
                    co_end = co_start + timedelta(seconds=co_sec)
                    results.append({
                        'Unit': u, 'Product': 'CHANGEOVER', 'Operation': 'SETUP',
                        'Quantity': 0, 'Time_Spent_Sec': co_sec,
                        'Start_Time': co_start, 'End_Time': co_end, 'Type': 'Setup'
                    })
                    last_info['time'] = co_end
                
                prod_start = last_info['time']
                spent_time_sec = q * unit_time
                prod_end = prod_start + timedelta(seconds=spent_time_sec)
                
                results.append({
                    'Unit': u, 'Product': p, 'Operation': o,
                    'Quantity': q, 'Time_Spent_Sec': spent_time_sec,
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
            max_workload = df_res.groupby('Unit')['Time_Spent_Sec'].sum().max()
        
        return df_res, max_workload, pd.DataFrame(unmet_results)
    else:
        return None, 0, pd.DataFrame()
