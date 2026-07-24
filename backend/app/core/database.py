from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session


def init_database() -> None:
    from app.models.analysis import EditDecisionGraphRecord, EditGraphRevision, ProjectAnalysis  # noqa: F401
    from app.models.memory import (  # noqa: F401
        DirectorMemoryEvidence,
        DirectorMemoryProfile,
        ProjectPerformanceSignal,
    )
    from app.models.project import Project, ProjectAsset  # noqa: F401

    Base.metadata.create_all(bind=engine)
