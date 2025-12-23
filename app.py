import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from core.optimizer import solve_production_allocation
import config.data_config as data_config
import yaml
import os

st.set_page_config(page_title="Production Schedule Dashboard", layout="wide")

# YAML 설정 로드
config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
db_defaults = {"user": "ADMIN", "password": "", "dsn": "localhost:1521/xe"}
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
        db_defaults.update(full_config.get('database', {}))

st.title("🏭 Production Allocation & Scheduling Dashboard")
st.markdown("""
이 대시보드는 최적화 엔진을 통해 계산된 장비별 작업 할당 결과를 시각화합니다. 
제품/공정 전환을 최소화하고 계획 달성을 최대화하는 할당안을 보여줍니다.
""")

# 1. 사이드바: 데이터 구성 확인
with st.sidebar:
    st.header("📋 Data Source")
    use_db_data = st.checkbox("Use Oracle DB Data", value=False)
    
    if use_db_data:
        st.warning("Make sure your SQL queries in `database/manager.py` are correct!")
    
    st.header("📋 Input Preview")
    if not use_db_data:
        st.subheader("Demands (Sample)")
        st.json(data_config.DEMAND)
        st.subheader("WIP (Sample)")
        st.json({str(k): v for k, v in data_config.WIP.items()})
        
        active_demand = data_config.DEMAND
        active_eqp = data_config.EQUIPMENT_MODELS
        active_proc = data_config.PROCESS_CONFIG
        active_avail = data_config.AVAILABLE_TIME
        active_wip = data_config.WIP
    else:
        # DB에서 데이터 가져오기 시도
        from database.manager import OracleManager
        mgr = OracleManager(db_defaults['user'], db_defaults['password'], db_defaults['dsn'])
        d, e, p, w = mgr.fetch_inputs()
        if d:
            st.success("Successfully loaded data from Oracle!")
            st.subheader("Demands (Oracle)")
            st.json(d)
            st.subheader("WIP (Oracle)")
            st.json({str(k): v for k, v in w.items()})
            active_demand, active_eqp, active_proc, active_wip = d, e, p, w
            active_avail = data_config.AVAILABLE_TIME
        else:
            st.error("Failed to load Oracle data. Using sample data instead.")
            active_demand, active_eqp, active_proc = data_config.DEMAND, data_config.EQUIPMENT_MODELS, data_config.PROCESS_CONFIG
            active_avail = data_config.AVAILABLE_TIME
            active_wip = data_config.WIP

# 2. 최적화 실행
if st.button("🚀 Run Optimizer"):
    with st.spinner("Calculating optimal schedule..."):
        df_results, bottleneck_time, df_unmet = solve_production_allocation(
            active_demand, active_eqp, active_proc, active_avail, wip=active_wip
        )
    
    if df_results is not None:
        st.success("Optimization Successfully Completed!")
        
        # 메트릭 표시
        col1, col2, col3 = st.columns(3)
        col1.metric("Bottleneck Time", f"{bottleneck_time:.0f}s")
        col2.metric("Line Efficiency", f"{(bottleneck_time/data_config.AVAILABLE_TIME)*100:.1f}%")
        col3.metric("Total Tasks", len(df_results))
        
        # 3. 간트 차트 (Gantt Chart)
        st.header("📅 Production Timeline (Gantt Chart)")
        
        # Plotly용 데이터 정리
        df_gantt = df_results.copy()
        df_gantt['Label'] = df_gantt['Product'] + " (" + df_gantt['Operation'] + ")"
        
        fig = px.timeline(
            df_gantt, 
            x_start="Start_Time", 
            x_end="End_Time", 
            y="Unit", 
            color="Product",
            hover_data=["Operation", "Quantity", "Time_Spent_Sec"],
            text="Label",
            title="Equipment Schedule Gantt Chart"
        )
        fig.update_yaxes(autorange="reversed") # 유닛 순서 유지
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. 미충족 수요 (Unmet Demand)
        if not df_unmet.empty:
            st.warning("⚠️ Unmet Demand Detected")
            st.table(df_unmet)
        else:
            st.info("✅ All demands are fully met.")
            
        # 5. 상세 데이터 테이블
        with st.expander("🔍 View Raw Allocation Data"):
            st.dataframe(df_results, use_container_width=True)
            
        # 6. 유닛별 요약
        st.header("📊 Unit Workload Summary")
        unit_summary = df_results.groupby('Unit')['Time_Spent_Sec'].sum().reset_index()
        fig_bar = px.bar(unit_summary, x='Unit', y='Time_Spent_Sec', title="Workload per Unit (Seconds)")
        st.plotly_chart(fig_bar, use_container_width=True)

        # 7. Oracle DB 적재 섹션
        st.divider()
        st.header("🗄️ Save Results to Oracle DB")
        with st.expander("Oracle Connection Settings"):
            db_user = st.text_input("User", value=db_defaults['user'])
            db_pwd = st.text_input("Password", type="password", value=db_defaults['password'])
            db_dsn = st.text_input("DSN", value=db_defaults['dsn'])

        if st.button("💾 Upload to Oracle"):
            from database.manager import OracleManager
            mgr = OracleManager(db_user, db_pwd, db_dsn)
            # Production 타입만 적재 (Changeover 제외)
            prod_only_df = df_results[df_results['Type'] == 'Production']
            mgr.upload_results(prod_only_df)
            st.success(f"Successfully uploaded {len(prod_only_df)} production records.")

    else:
        st.error("Optimization Failed. Please check the constraints.")

else:
    st.info("위의 버튼을 눌러 최적화를 시작하세요.")
