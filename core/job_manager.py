import uuid
import logging
import yaml
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from core.optimizer import solve_production_allocation
from database.manager import OracleManager
import config.data_config as data_config

logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        self.load_config()
        
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.jobs = {} # {job_id: {"status": ..., "result": ..., "mode": ...}}
        
        # 스케줄러 설정
        self.scheduler = BackgroundScheduler()
        if self.sched_enabled:
            self.scheduler.add_job(self.submit_job, 'interval', minutes=self.sched_interval, id='batch_prod')
        self.scheduler.start()

    def load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            conf = yaml.safe_load(f)
        self.max_workers = conf.get('api', {}).get('workers', 2)
        self.timeout = conf.get('optimization', {}).get('timeout_sec', 600)
        self.sched_enabled = conf.get('scheduler', {}).get('enabled', False)
        self.sched_interval = conf.get('scheduler', {}).get('interval_min', 60)
        self.system_mode = conf.get('system_mode', 'local_test')

    def generate_job_id(self):
        return str(uuid.uuid4())

    def _run_task(self, job_id, mode):
        try:
            self.jobs[job_id]["status"] = "RUNNING"
            self.jobs[job_id]["start_time"] = datetime.now()
            
            # 지정된 모드로 매니저 초기화
            mgr = OracleManager(mode=mode)
            demands, eqp_models, proc_config, wip = mgr.fetch_inputs()
            
            if demands is None:
                self.jobs[job_id]["status"] = "FAILED"
                self.jobs[job_id]["error"] = "Failed to fetch inputs"
                return

            df_results, b_time, df_unmet = solve_production_allocation(
                demands=demands,
                eqp_models=eqp_models,
                proc_config=proc_config,
                avail_time=data_config.AVAILABLE_TIME,
                wip=wip
            )
            
            if df_results is not None:
                prod_only_df = df_results[df_results['Type'] == 'Production']
                mgr.upload_results(prod_only_df)
                
                self.jobs[job_id].update({
                    "status": "COMPLETED",
                    "result": {"bottleneck": float(b_time), "records": len(prod_only_df)},
                    "end_time": datetime.now()
                })
            else:
                self.jobs[job_id]["status"] = "FAILED"
                self.jobs[job_id]["error"] = "Optimization Infeasible"

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.jobs[job_id]["status"] = "FAILED"
            self.jobs[job_id]["error"] = str(e)

    def submit_job(self, mode=None):
        job_id = self.generate_job_id()
        target_mode = mode or self.system_mode
        self.jobs[job_id] = {
            "status": "PENDING",
            "submit_time": datetime.now(),
            "mode": target_mode
        }
        self.executor.submit(self._run_task, job_id, target_mode)
        return job_id

    def get_job_status(self, job_id):
        return self.jobs.get(job_id)

    def update_system_config(self, mode=None, sched_enabled=None, sched_interval=None):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            conf = yaml.safe_load(f)
        
        if mode: conf['system_mode'] = mode
        if sched_enabled is not None: conf['scheduler']['enabled'] = sched_enabled
        if sched_interval: conf['scheduler']['interval_min'] = sched_interval
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(conf, f)
        
        self.load_config()
        # 스케줄러 갱신
        self.scheduler.remove_all_jobs()
        if self.sched_enabled:
            self.scheduler.add_job(self.submit_job, 'interval', minutes=self.sched_interval, id='batch_prod')
        return conf
