from core.optimizer import solve_production_allocation
from database.manager import OracleManager
import config.data_config as data_config
import pandas as pd

def main():
    print("=== Production Allocation Verification ===")
    
    # 1. DB 매니저 초기화 (config.yaml의 system_mode 기준)
    mgr = OracleManager()
    
    # 2. 데이터 가져오기 (local_test면 샘플, 그 외엔 DB)
    print(f"[1/3] Fetching data (Mode: {mgr.mode})...")
    demands, eqp_models, proc_config, wip, eqp_wip, tools = mgr.fetch_inputs()
    
    if demands is None:
        print("!!! Error: Failed to fetch inputs.")
        return
        
    avail_time = data_config.AVAILABLE_TIME
    
    # 3. 최적화 실행
    print("[2/3] Solving optimization problem...")
    df_results, bottleneck_time, df_unmet = solve_production_allocation(
        demands=demands,
        eqp_models=eqp_models,
        proc_config=proc_config,
        avail_time=avail_time,
        wip=wip,
        eqp_wip=eqp_wip,
        tools=tools
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
