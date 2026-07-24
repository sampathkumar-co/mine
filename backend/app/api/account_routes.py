from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.platform import User, WorkspaceMembership
from app.schemas.operations import AccountContextRead
from app.schemas.platform import UserRead, WorkspaceRead
from app.services.email import email_is_verified

router = APIRouter()


def _current_user_id(request: Request) -> UUID:
    user_id = getattr(request.state, "user_id", None)
    if not isinstance(user_id, UUID):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user_id


@router.get("/auth/account", response_model=AccountContextRead, tags=["authentication"])
def get_account_context(
    request: Request,
    db: Session = Depends(get_db),
) -> AccountContextRead:
    user = db.get(User, _current_user_id(request))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    memberships = list(
        db.scalars(
            select(WorkspaceMembership)
            .where(WorkspaceMembership.user_id == user.id)
            .order_by(WorkspaceMembership.created_at)
        ).all()
    )
    return AccountContextRead(
        user=UserRead(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            email_verified=email_is_verified(db, user.id),
            created_at=user.created_at,
        ),
        workspaces=[
            WorkspaceRead(
                id=item.workspace.id,
                name=item.workspace.name,
                slug=item.workspace.slug,
                role=item.role,
                created_at=item.workspace.created_at,
            )
            for item in memberships
        ],
    )
