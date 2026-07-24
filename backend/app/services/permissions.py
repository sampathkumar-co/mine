from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.platform import WorkspaceMembership

ROLE_RANK = {
    "viewer": 10,
    "editor": 20,
    "admin": 30,
    "owner": 40,
}
VALID_ROLES = frozenset(ROLE_RANK)


def normalize_role(role: str) -> str:
    normalized = role.strip().casefold()
    if normalized not in VALID_ROLES:
        raise ValueError(f"Unsupported workspace role: {role}")
    return normalized


def membership_for(db: Session, workspace_id: UUID, user_id: UUID) -> WorkspaceMembership | None:
    return db.scalar(
        select(WorkspaceMembership).where(
            WorkspaceMembership.workspace_id == workspace_id,
            WorkspaceMembership.user_id == user_id,
        )
    )


def has_role(role: str, minimum_role: str) -> bool:
    return ROLE_RANK.get(role, 0) >= ROLE_RANK[minimum_role]


def can_read(role: str) -> bool:
    return has_role(role, "viewer")


def can_edit(role: str) -> bool:
    return has_role(role, "editor")


def can_manage_members(role: str) -> bool:
    return has_role(role, "admin")


def can_manage_billing(role: str) -> bool:
    return role == "owner"
