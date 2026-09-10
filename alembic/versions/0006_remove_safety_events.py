"""Remove the retired moderation audit table.

Revision ID: c6d7e8f9a012
Revises: b5f23d1a8c41
Create Date: 2026-09-09
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c6d7e8f9a012"
down_revision: str | None = "b5f23d1a8c41"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_table("safety_events", schema="elysia")


def downgrade() -> None:
    op.create_table(
        "safety_events",
        sa.Column("request_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=True),
        sa.Column(
            "direction",
            sa.Enum("input", "output", name="safety_direction", native_enum=False),
            nullable=False,
        ),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("severity", sa.String(length=20), nullable=True),
        sa.Column("action", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["elysia.conversations.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["elysia.users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        schema="elysia",
    )
    for column in ("conversation_id", "created_at", "request_id", "user_id"):
        op.create_index(
            f"ix_safety_events_{column}",
            "safety_events",
            [column],
            unique=False,
            schema="elysia",
        )
