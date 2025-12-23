import streamlit as st
import requests
import pandas as pd
import time
import yaml
import os

st.set_page_config(page_title="Production Balancer Admin", layout="wide")

API_URL = "http://localhost:8000"

st.title("⚙️ Production Balancer Admin Panel")

# 1. 시스템 설정 관리
st.header("🌐 System Environment & Scheduler")
config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')

with open(config_path, 'r', encoding='utf-8') as f:
    conf = yaml.safe_load(f)

col1, col2, col3 = st.columns(3)

with col1:
    new_mode = st.selectbox("Current Mode", ["production", "development", "local_test"], 
                            index=["production", "development", "local_test"].index(conf['system_mode']))

with col2:
    sched_on = st.toggle("Enable Batch Scheduler", value=conf['scheduler']['enabled'])

with col3:
    interval = st.number_input("Batch Interval (min)", min_value=1, value=conf['scheduler']['interval_min'])

if st.button("Apply Changes"):
    # API를 통해 설정을 변경하거나 직접 파일을 수정 (여기서는 직접 수정 후 JobManager 리로드 유도 - 실제는 API 엔드포인트 호출 추천)
    conf['system_mode'] = new_mode
    conf['scheduler']['enabled'] = sched_on
    conf['scheduler']['interval_min'] = interval
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(conf, f)
    st.success("Settings applied! (Please restart API server if scheduler doesn't update immediately)")

st.divider()

# 2. 작업 수동 실행 및 큐 모니터링
st.header("📊 Job Queue & Manual Trigger")

if st.button("🚀 Trigger Manual Job Now"):
    try:
        res = requests.post(f"{API_URL}/run-optimization")
        if res.status_code == 200:
            st.success(f"Job submitted! ID: {res.json()['job_id']}")
        else:
            st.error("Failed to trigger job")
    except:
        st.error("Is API server running?")

# 큐 리스트 조회 (API에 구현 필요)
st.subheader("Active Jobs (Last 10)")
try:
    res = requests.get(f"{API_URL}/jobs")
    if res.status_code == 200:
        jobs_data = res.json()
        if jobs_data:
            df = pd.DataFrame.from_dict(jobs_data, orient='index')
            st.table(df[['status', 'mode', 'submit_time', 'start_time', 'end_time']].sort_values('submit_time', ascending=False))
        else:
            st.info("No jobs found.")
except:
    st.info("Start the API server to see live logs.")

if st.button("Refresh List"):
    st.rerun()
