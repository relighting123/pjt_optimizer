from core.optimizer import solve_production_allocation
from database.manager import OracleManager
import config.data_config as data_config
import pandas as pd

def main():
    print("=== Production Allocation Verification (Sample Data) ===")
    
    # [Verification Mode] DB 접근 없이 data_config의 테스트 케이스 사용
    demands = data_config.DEMAND
    eqp_models = data_config.EQUIPMENT_MODELS
    proc_config = data_config.PROCESS_CONFIG
    wip = data_config.WIP
    avail_time = data_config.AVAILABLE_TIME
    
    print(f"Testing with: {len(demands)} Products, {len(eqp_models)} Models")
    
    # 3. 최적화 실행
    print("[2/3] Solving optimization problem...")
    df_results, bottleneck_time, df_unmet = solve_production_allocation(
        demands=demands,
        eqp_models=eqp_models,
        proc_config=proc_config,
        avail_time=avail_time,
        wip=wip
    )
    
    if df_results is not None:
        print(f"Success! Bottleneck Time: {bottleneck_time:.2f}s")
        print("\n--- Detailed Allocation Result ---")
        print(df_results[['Unit', 'Product', 'Operation', 'Quantity', 'Type']])
        
        if not df_unmet.empty:
            print("\n!!! WARNING: UNMET DEMAND !!!")
            print(df_unmet)
            
        print("\n=== Verification Completed ===")
    else:
        print("Optimization Failed.")

if __name__ == "__main__":
    main()
