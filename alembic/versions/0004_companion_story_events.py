"""add durable companion story events

Revision ID: a94ce67c152d
Revises: 3f8d8c4f7a21
Create Date: 2026-09-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a94ce67c152d"
down_revision: str | None = "3f8d8c4f7a21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "story_events",
        sa.Column("companion_id", sa.UUID(), nullable=False),
        sa.Column("external_event_id", sa.String(length=200), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("happened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["companion_id"],
            ["elysia.companions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "companion_id",
            "external_event_id",
            name="uq_story_event_companion_external",
        ),
        schema="elysia",
    )
    op.create_index(
        "ix_story_events_companion_happened_at",
        "story_events",
        ["companion_id", "happened_at"],
        schema="elysia",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_story_events_companion_happened_at",
        table_name="story_events",
        schema="elysia",
    )
    op.drop_table("story_events", schema="elysia")
