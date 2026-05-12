import os
import asyncio
from celery import Celery
from main_minimal import (
    store,
    QueryRequest,
    _async_generate_chunk
)

celery_app = Celery(
    "qna_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # This ensures a worker doesn't get stuck forever if an LLM call hangs
    task_time_limit=10800, 
)

@celery_app.task(bind=True)
def generated_questions_task(self, session_id: str, request_dict: dict, chunk_size: int = 2):
    """
    This runs in the background. We wrap the async logic inside asyncio.run()
    because Celery workers are synchronous by default.
    """
    req = QueryRequest(**request_dict)
    store.update_session_status(session_id, "running")
    try:
        # We call the generation logic from main_minimal. 
        # req.num_questions is already set to the chunk size!
        asyncio.run(_async_generate_chunk(req, session_id))
        
        # Check if this was the last chunk to mark session as completed
        session_data = store.get_session(session_id)
        if session_data and session_data["total_generated"] >= session_data["total_requested"]:
            store.update_session_status(session_id, "completed")
            
    except Exception as e:
        print(f"Task Failed: {e}")
        store.update_session_status(session_id, "failed")

