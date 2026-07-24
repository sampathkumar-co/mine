from collections.abc import Generator
from pathlib import Path

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


def import_models() -> None:
    import app.models.analysis  # noqa: F401
    import app.models.camera  # noqa: F401
    import app.models.governance  # noqa: F401
    import app.models.memory  # noqa: F401
    import app.models.operations  # noqa: F401
    import app.models.platform  # noqa: F401
    import app.models.project  # noqa: F401
    import app.models.subscriptions  # noqa: F401


def init_database() -> None:
    import_models()
    Base.metadata.create_all(bind=engine)


def migrate_database(revision: str = "head") -> None:
    from alembic.config import Config

    from alembic import command

    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    config = Config(str(config_path))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(config, revision)
