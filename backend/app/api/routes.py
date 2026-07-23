from fastapi import APIRouter, status

from app.schemas.projects import ProjectAccepted, ProjectCreate
from app.worker.tasks import run_project_pipeline

router = APIRouter()


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "director-os-api"}


@router.post(
    "/projects",
    response_model=ProjectAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["projects"],
)
def create_project(payload: ProjectCreate) -> ProjectAccepted:
    accepted = ProjectAccepted()
    run_project_pipeline.delay(str(accepted.project_id), payload.contract.model_dump())
    return accepted
