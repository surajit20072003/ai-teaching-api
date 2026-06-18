import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, Integer, Float, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://teaching_user:teaching_pass@postgres:5432/teaching_db")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=25,        # base pool (was 10)
    max_overflow=50,     # extra burst connections (was 20) → total 75
    pool_timeout=30,     # wait 30s for a connection before error
    pool_pre_ping=True,  # detect dead connections automatically
)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class TeachingCache(Base):
    __tablename__ = "teaching_qa_cache"

    id                     = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    question_hash          = Column(String(64), nullable=False)
    question_text          = Column(Text, nullable=False)
    subject_id             = Column(String, nullable=True)
    topic_id               = Column(String, nullable=True)
    chapter_id             = Column(String, nullable=True)
    language               = Column(String(10), default="hi-IN")
    variation_number       = Column(Integer, default=1)
    presentation_slides    = Column(JSONB, default=list)
    latex_formulas         = Column(JSONB, default=list)
    slide_audio_urls       = Column(JSONB, default=dict)
    total_duration_seconds = Column(Float, default=0.0)
    usage_count            = Column(Integer, default=0)
    created_at             = Column(TIMESTAMP(timezone=True), server_default=func.now())
    question_embedding     = Column(Vector(384), nullable=True)   # all-MiniLM-L6-v2 dims

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
