from core.optimizer import solve_production_allocation
from database.manager import OracleManager
import config.data_config as data_config
import pandas as pd

def main():
    print("=== Automated Production Allocation Flow ===")
    
    # 1. DB 매니저 초기화
    mgr = OracleManager()
    
    # 2. 실데이터 가져오기 시도
    print("[1/3] Fetching real data from Oracle...")
    demands, eqp_models, proc_config = mgr.fetch_inputs()
    
    # DB 데이터가 없을 경우 샘플 데이터 사용 유도
    if not demands:
        print("!!! Warning: Real data not found. Using sample data for demonstration.")
        demands = data_config.DEMAND
        eqp_models = data_config.EQUIPMENT_MODELS
        proc_config = data_config.PROCESS_CONFIG
        wip = data_config.WIP
    else:
        wip = data_config.WIP
    
    # 3. 최적화 실행
    print("[2/3] Solving optimization problem...")
    df_results, bottleneck_time, df_unmet = solve_production_allocation(
        demands=demands,
        eqp_models=eqp_models,
        proc_config=proc_config,
        avail_time=data_config.AVAILABLE_TIME,
        wip=wip
    )
    
    if df_results is not None:
        print(f"Success! Bottleneck Time: {bottleneck_time:.2f}s")
        
        # 4. 결과 적재
        print("[3/3] Uploading results to Oracle...")
        prod_only_df = df_results[df_results['Type'] == 'Production']
        mgr.upload_results(prod_only_df)
        
        print("\n=== Automation Sequence Completed ===")
    else:
        print("Optimization Failed.")

if __name__ == "__main__":
    main()
