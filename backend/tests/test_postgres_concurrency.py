from __future__ import annotations

import os
import threading
from uuid import uuid4

import pytest
from sqlalchemy import select, text

from app.core.config import get_settings
from app.core.database import Base, SessionLocal, engine, import_models
from app.core.enums import ProjectStatus
from app.models.operations import AuthSessionRecord, ProductionJob
from app.models.platform import User, Workspace, WorkspaceMembership
from app.models.project import Project
from app.services.jobs import JobConflictError, enqueue_project_job, enqueue_revision_job
from app.services.sessions import SessionError, issue_session, rotate_session

pytestmark = pytest.mark.skipif(
    os.getenv("DIRECTOR_RUN_POSTGRES_TESTS") != "1",
    reason="Set DIRECTOR_RUN_POSTGRES_TESTS=1 for PostgreSQL concurrency tests.",
)
settings = get_settings()


def _reset_public_schema() -> None:
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))


@pytest.fixture(autouse=True)
def reset_postgres_schema():
    import_models()
    _reset_public_schema()
    Base.metadata.create_all(bind=engine)
    yield
    _reset_public_schema()


def _identity() -> tuple[User, Workspace, Project]:
    with SessionLocal() as db:
        user = User(
            email=f"concurrency-{uuid4()}@example.com",
            display_name="Concurrency",
            password_hash="not-used",
        )
        db.add(user)
        db.flush()
        workspace = Workspace(
            name="Concurrency Studio",
            slug=f"concurrency-{uuid4().hex[:8]}",
            created_by_user_id=user.id,
        )
        db.add(workspace)
        db.flush()
        db.add(
            WorkspaceMembership(
                workspace_id=workspace.id,
                user_id=user.id,
                role="owner",
            )
        )
        project = Project(
            user_id=user.id,
            workspace_id=workspace.id,
            status=ProjectStatus.READY_TO_QUEUE,
            contract={"objective": "Concurrency test"},
        )
        db.add(project)
        db.commit()
        db.refresh(user)
        db.refresh(workspace)
        db.refresh(project)
        db.expunge(user)
        db.expunge(workspace)
        db.expunge(project)
        return user, workspace, project


def test_refresh_rotation_is_atomic_under_concurrency() -> None:
    user, _, _ = _identity()
    with SessionLocal() as db:
        attached = db.get(User, user.id)
        assert attached is not None
        issued = issue_session(db, attached, settings)
        db.commit()
        refresh_token = issued.refresh_token
        csrf_token = issued.csrf_token
        family_id = issued.record.family_id

    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def rotate() -> None:
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            try:
                rotate_session(db, refresh_token, csrf_token, settings)
                db.commit()
                result = "success"
            except SessionError:
                db.commit()
                result = "reuse"
            with lock:
                outcomes.append(result)

    threads = [threading.Thread(target=rotate), threading.Thread(target=rotate)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()

    assert sorted(outcomes) == ["reuse", "success"]
    with SessionLocal() as db:
        family = list(
            db.scalars(
                select(AuthSessionRecord).where(AuthSessionRecord.family_id == family_id)
            ).all()
        )
        assert len(family) == 2
        assert all(record.revoked_at is not None for record in family)


def test_concurrent_project_start_creates_one_job() -> None:
    user, _, project = _identity()
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def start() -> None:
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            try:
                enqueue_project_job(
                    db,
                    project.id,
                    kind="production",
                    allowed_statuses={ProjectStatus.READY_TO_QUEUE},
                    created_by_user_id=user.id,
                )
                db.commit()
                result = "success"
            except JobConflictError:
                db.rollback()
                result = "conflict"
            with lock:
                outcomes.append(result)

    threads = [threading.Thread(target=start), threading.Thread(target=start)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()

    assert sorted(outcomes) == ["conflict", "success"]
    with SessionLocal() as db:
        jobs = list(
            db.scalars(select(ProductionJob).where(ProductionJob.project_id == project.id)).all()
        )
        assert len(jobs) == 1


def test_concurrent_revision_enqueue_keeps_one_active_job() -> None:
    user, _, project = _identity()
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def enqueue(version: int) -> None:
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            try:
                enqueue_revision_job(
                    db,
                    project.id,
                    version,
                    created_by_user_id=user.id,
                )
                db.commit()
                result = "success"
            except JobConflictError:
                db.rollback()
                result = "conflict"
            with lock:
                outcomes.append(result)

    threads = [
        threading.Thread(target=enqueue, args=(2,)),
        threading.Thread(target=enqueue, args=(3,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()

    assert sorted(outcomes) == ["conflict", "success"]
    with SessionLocal() as db:
        jobs = list(
            db.scalars(
                select(ProductionJob).where(
                    ProductionJob.project_id == project.id,
                    ProductionJob.kind == "revision",
                )
            ).all()
        )
        assert len(jobs) == 1
