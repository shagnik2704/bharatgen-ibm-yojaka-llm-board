from sqlalchemy import Column, String, Float, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid
import json
import os
from pathlib import Path

def _prepare_database_url() -> str:
    """Prepare a usable DATABASE_URL and create parent folders for sqlite files."""
    raw_url = os.getenv("DATABASE_URL")

    if not raw_url:
        default_db = Path(__file__).resolve().parent / "data" / "bharatgen_questions.db"
        default_db.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{default_db.as_posix()}"

    if raw_url.startswith("sqlite:///") and raw_url != "sqlite:///:memory:":
        db_path_str = raw_url[len("sqlite:///"):]
        if db_path_str:
            db_path = Path(db_path_str)
            if not db_path.is_absolute():
                db_path = (Path(__file__).resolve().parent / db_path).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path.as_posix()}"

    return raw_url


DATABASE_URL = _prepare_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args
)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class QuestionDB(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    question = Column(Text)
    answer = Column(Text)

    alignment_score = Column(Float)

    # searchable field
    model_id = Column(String)

    # full blobs
    req_json = Column(Text)
    scores_json = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)


def save_question(req, q, scores, alignment):

    db = SessionLocal()

    row = QuestionDB(
        question=q.get("question"),
        answer=q.get("answer"),
        alignment_score=alignment,
        model_id=req.model_id,

        req_json=json.dumps(req.dict()),
        scores_json=json.dumps(scores),
    )

    db.add(row)
    db.commit()
    db.close()


# Create tables
Base.metadata.create_all(bind=engine)
