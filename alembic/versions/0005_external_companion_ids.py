"""Store external companion IDs without a local-companions foreign key."""

from collections.abc import Sequence

from alembic import op

revision: str = "b5f23d1a8c41"
down_revision: str | None = "a94ce67c152d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = (
    "conversations",
    "relationship_context",
    "memories",
    "ai_events",
    "media_assets",
    "story_events",
)


def upgrade() -> None:
    for table in _TABLES:
        op.execute(
            f"ALTER TABLE elysia.{table} DROP CONSTRAINT IF EXISTS {table}_companion_id_fkey"
        )


def downgrade() -> None:
    # A downgrade is unsafe: current external IDs may not exist in the retired
    # local catalogue, so restoring these constraints could reject valid data.
    raise RuntimeError("Cannot restore local companion foreign keys safely.")
