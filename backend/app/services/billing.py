from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.operations import (
    BillingAccount,
    BillingEntry,
    ProjectBillingReservation,
    UserStarterCreditGrant,
)
from app.models.platform import User, Workspace
from app.models.project import Project

CREDIT_QUANTUM = Decimal("0.0001")


class BillingError(ValueError):
    pass


class InsufficientCreditsError(BillingError):
    pass


def credits(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(CREDIT_QUANTUM, rounding=ROUND_HALF_UP)


def ensure_billing_account(
    db: Session,
    workspace_id: UUID,
    *,
    starter_credits: Decimal | int | float | str = 0,
) -> BillingAccount:
    account = db.get(BillingAccount, workspace_id)
    if account is None:
        account = BillingAccount(
            workspace_id=workspace_id,
            balance_credits=credits(starter_credits),
            reserved_credits=credits(0),
        )
        db.add(account)
        db.flush()
    return account




def ensure_workspace_billing(
    db: Session,
    workspace_id: UUID,
    settings,
) -> BillingAccount:
    account = ensure_billing_account(db, workspace_id)
    if settings.starter_grants_per_user <= 0 or settings.starter_credits <= 0:
        return account
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        return account
    db.scalar(select(User).where(User.id == workspace.created_by_user_id).with_for_update())
    grant = db.get(UserStarterCreditGrant, workspace.created_by_user_id)
    if grant is not None:
        return account
    post_adjustment(
        db,
        workspace_id,
        settings.starter_credits,
        idempotency_key=f"user:{workspace.created_by_user_id}:starter-grant",
        description="One-time starter credits granted to the account owner",
        actor_user_id=workspace.created_by_user_id,
        kind="grant",
    )
    db.add(
        UserStarterCreditGrant(
            user_id=workspace.created_by_user_id,
            workspace_id=workspace_id,
            amount_credits=credits(settings.starter_credits),
        )
    )
    db.flush()
    return account


def _locked_account(db: Session, workspace_id: UUID) -> BillingAccount:
    account = db.scalar(
        select(BillingAccount)
        .where(BillingAccount.workspace_id == workspace_id)
        .with_for_update()
    )
    if account is None:
        account = ensure_billing_account(db, workspace_id)
    return account


def _entry_exists(db: Session, idempotency_key: str) -> BillingEntry | None:
    return db.scalar(
        select(BillingEntry).where(BillingEntry.idempotency_key == idempotency_key)
    )


def post_adjustment(
    db: Session,
    workspace_id: UUID,
    amount: Decimal | int | float | str,
    *,
    idempotency_key: str,
    description: str,
    actor_user_id: UUID | None = None,
    project_id: UUID | None = None,
    kind: str = "adjustment",
    metadata: dict[str, object] | None = None,
) -> BillingEntry:
    existing = _entry_exists(db, idempotency_key)
    if existing is not None:
        return existing
    normalized = credits(amount)
    account = _locked_account(db, workspace_id)
    if account.balance_credits + normalized < account.reserved_credits:
        raise InsufficientCreditsError("Adjustment would reduce available credits below zero")
    account.balance_credits = credits(account.balance_credits + normalized)
    entry = BillingEntry(
        workspace_id=workspace_id,
        project_id=project_id,
        actor_user_id=actor_user_id,
        kind=kind,
        amount_credits=normalized,
        idempotency_key=idempotency_key,
        description=description[:500],
        entry_metadata=metadata or {},
    )
    db.add(entry)
    db.flush()
    return entry


def estimate_project_credits(project: Project) -> Decimal:
    duration = Decimal(str(project.contract.get("target_duration_seconds", 45)))
    source_count = sum(1 for asset in project.assets if asset.kind.value == "source_video")
    tier = Decimal(str(project.contract.get("tier", 1)))
    estimate = Decimal("0.75") + duration * Decimal("0.035")
    estimate += Decimal(max(1, source_count)) * Decimal("0.20")
    estimate *= Decimal("1") + (tier - Decimal("1")) * Decimal("0.15")
    return max(credits("1"), credits(estimate))


def reserve_project_credits(
    db: Session,
    project: Project,
    *,
    actor_user_id: UUID | None,
) -> ProjectBillingReservation:
    if project.workspace_id is None:
        raise BillingError("Project has no billing workspace")
    existing = db.get(ProjectBillingReservation, project.id)
    if existing is not None:
        if existing.status in {"reserved", "settled"}:
            return existing
        raise BillingError("Project billing reservation is not reusable")

    estimate = estimate_project_credits(project)
    account = _locked_account(db, project.workspace_id)
    available = credits(account.balance_credits - account.reserved_credits)
    if available < estimate:
        raise InsufficientCreditsError(
            f"Project requires about {estimate} credits; {available} are available"
        )
    account.reserved_credits = credits(account.reserved_credits + estimate)
    reservation = ProjectBillingReservation(
        project_id=project.id,
        workspace_id=project.workspace_id,
        reserved_credits=estimate,
        status="reserved",
    )
    db.add(reservation)
    db.add(
        BillingEntry(
            workspace_id=project.workspace_id,
            project_id=project.id,
            actor_user_id=actor_user_id,
            kind="reserve",
            amount_credits=estimate,
            idempotency_key=f"project:{project.id}:reserve",
            description="Reserved credits for autonomous production",
            entry_metadata={"estimated_credits": str(estimate)},
        )
    )
    db.flush()
    return reservation


def settle_project_credits(
    db: Session,
    project: Project,
    *,
    actual_credits: Decimal | int | float | str | None = None,
) -> ProjectBillingReservation | None:
    reservation = db.get(ProjectBillingReservation, project.id)
    if reservation is None or reservation.status == "settled":
        return reservation
    if reservation.status != "reserved":
        return reservation
    account = _locked_account(db, reservation.workspace_id)
    actual = credits(actual_credits if actual_credits is not None else reservation.reserved_credits)
    actual = min(actual, credits(reservation.reserved_credits))
    account.reserved_credits = max(
        credits(0),
        credits(account.reserved_credits - reservation.reserved_credits),
    )
    account.balance_credits = credits(account.balance_credits - actual)
    reservation.settled_credits = actual
    reservation.status = "settled"
    if _entry_exists(db, f"project:{project.id}:settle") is None:
        db.add(
            BillingEntry(
                workspace_id=reservation.workspace_id,
                project_id=project.id,
                kind="settle",
                amount_credits=-actual,
                idempotency_key=f"project:{project.id}:settle",
                description="Settled autonomous production usage",
                entry_metadata={
                    "reserved_credits": str(reservation.reserved_credits),
                    "settled_credits": str(actual),
                },
            )
        )
    db.flush()
    return reservation


def release_project_reservation(
    db: Session,
    project: Project,
    *,
    reason: str,
) -> ProjectBillingReservation | None:
    reservation = db.get(ProjectBillingReservation, project.id)
    if reservation is None or reservation.status != "reserved":
        return reservation
    account = _locked_account(db, reservation.workspace_id)
    account.reserved_credits = max(
        credits(0),
        credits(account.reserved_credits - reservation.reserved_credits),
    )
    reservation.status = "released"
    if _entry_exists(db, f"project:{project.id}:release") is None:
        db.add(
            BillingEntry(
                workspace_id=reservation.workspace_id,
                project_id=project.id,
                kind="release",
                amount_credits=reservation.reserved_credits,
                idempotency_key=f"project:{project.id}:release",
                description=reason[:500],
                entry_metadata={"released_credits": str(reservation.reserved_credits)},
            )
        )
    db.flush()
    return reservation
