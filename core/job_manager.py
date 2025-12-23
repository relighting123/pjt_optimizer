import uuid
import time
import logging
import asyncio
import yaml
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from core.optimizer import solve_production_allocation
from database.manager import OracleManager
import config.data_config as data_config

logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self):
        # 설정 로드
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            conf = yaml.safe_load(f)
        
        self.max_workers = conf.get('api', {}).get('workers', 2)
        self.timeout = conf.get('optimization', {}).get('timeout_sec', 600)
        
        # 스레드 풀 및 작업 상태 저장소
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.jobs = {} # {job_id: {"status": ..., "result": ..., "start_time": ...}}

    def generate_job_id(self):
        return str(uuid.uuid4())

    def _run_task(self, job_id):
        """실제 최적화 업무 로직 (스레드에서 실행)"""
        try:
            self.jobs[job_id]["status"] = "RUNNING"
            self.jobs[job_id]["start_time"] = datetime.now()
            
            # 1. DB Fetch
            mgr = OracleManager()
            demands, eqp_models, proc_config, wip_db = mgr.fetch_inputs()
            
            if demands is None:
                # DB 실패시 샘플 데이터 (테스트용)
                demands = data_config.DEMAND
                eqp_models = data_config.EQUIPMENT_MODELS
                proc_config = data_config.PROCESS_CONFIG
                wip = data_config.WIP
            else:
                wip = data_config.WIP # DB 연동 후 fetch_inputs()에서 같이 가져오도록 확장 가능

            # 2. Optimize
            df_results, b_time, df_unmet = solve_production_allocation(
                demands=demands,
                eqp_models=eqp_models,
                proc_config=proc_config,
                avail_time=data_config.AVAILABLE_TIME,
                wip=wip
            )
            
            if df_results is not None:
                # 3. Upload
                prod_only_df = df_results[df_results['Type'] == 'Production']
                mgr.upload_results(prod_only_df)
                
                self.jobs[job_id].update({
                    "status": "COMPLETED",
                    "result": {
                        "bottleneck_time": float(b_time),
                        "records": len(prod_only_df)
                    },
                    "end_time": datetime.now()
                })
            else:
                self.jobs[job_id]["status"] = "FAILED"
                self.jobs[job_id]["error"] = "Infeasible result"

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.jobs[job_id]["status"] = "FAILED"
            self.jobs[job_id]["error"] = str(e)

    def submit_job(self):
        job_id = self.generate_job_id()
        self.jobs[job_id] = {
            "status": "PENDING",
            "submit_time": datetime.now()
        }
        
        # 스레드 풀에 작업 제출
        future = self.executor.submit(self._run_task, job_id)
        
        # 타임아웃 감시 로직 (별도 스레드나 비동기로 처리 가능하지만 여기서는 상태로 체크)
        return job_id

    def get_job_status(self, job_id):
        if job_id not in self.jobs:
            return None
        
        job_info = self.jobs[job_id]
        
        # 타임아웃 체크 로직
        if job_info["status"] == "RUNNING":
            elapsed = (datetime.now() - job_info["start_time"]).total_seconds()
            if elapsed > self.timeout:
                job_info["status"] = "TIMEOUT"
                job_info["error"] = f"Job exceeded time limit of {self.timeout}s"
        
        return job_info
