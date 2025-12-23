from fastapi import FastAPI, HTTPException
from core.job_manager import JobManager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Production Balancer API (Async Queue)", description="Queue-based Production Allocation System")
job_manager = JobManager()

@app.post("/run-optimization")
async def run_optimization():
    """
    작업을 큐에 등록하고 job_id를 반환합니다.
    """
    try:
        job_id = job_manager.submit_job()
        logger.info(f"Job submitted: {job_id}")
        return {
            "status": "ACCEPTED",
            "job_id": job_id,
            "message": "Optimization task has been queued."
        }
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job-status/{job_id}")
async def get_status(job_id: str):
    """
    job_id를 사용하여 현재 작업의 진행 상태를 확인합니다.
    """
    status_info = job_manager.get_job_status(job_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    return status_info

@app.get("/jobs")
async def get_all_jobs():
    """모든 작업의 상태 리스트를 반환합니다."""
    return job_manager.jobs

@app.get("/config")
async def get_queue_config():
    """현재 큐 및 워커 설정 정보를 확인합니다."""
    return {
        "max_workers": job_manager.max_workers,
        "timeout_sec": job_manager.timeout
    }

@app.get("/health")
async def health_check():
    return {"status": "UP"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
