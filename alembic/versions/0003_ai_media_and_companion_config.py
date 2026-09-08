"""add AI media architecture and companion voice configuration

Revision ID: 3f8d8c4f7a21
Revises: 14d5d2b06e4b
Create Date: 2026-09-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "3f8d8c4f7a21"
down_revision: str | None = "14d5d2b06e4b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "companions",
        sa.Column(
            "voice_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        schema="elysia",
    )

    # Preserve the existing companion UUID and every FK that points at it.
    # Installations created before the product rename may use either spelling.
    op.execute(
        """
        UPDATE elysia.companions
        SET slug = 'lina', name = 'Lina', version = version + 1
        WHERE slug IN ('anastacia', 'anastasia')
          AND NOT EXISTS (
              SELECT 1 FROM elysia.companions current_lina
              WHERE current_lina.slug = 'lina'
          )
        """
    )
    op.execute(
        """
        UPDATE elysia.companions legacy
        SET active = false
        WHERE legacy.slug IN ('anastacia', 'anastasia')
          AND EXISTS (
              SELECT 1 FROM elysia.companions current_lina
              WHERE current_lina.slug = 'lina'
          )
        """
    )

    # Match MemoryService's key-based upsert with a database guarantee.
    # Older duplicate rows are reduced to the most recently updated row.
    op.execute(
        """
        DELETE FROM elysia.memories older
        USING elysia.memories newer
        WHERE older.user_id = newer.user_id
          AND older.companion_id = newer.companion_id
          AND older.key = newer.key
          AND (older.updated_at, older.id) < (newer.updated_at, newer.id)
        """
    )
    op.create_unique_constraint(
        "uq_memory_user_companion_key",
        "memories",
        ["user_id", "companion_id", "key"],
        schema="elysia",
    )
    op.drop_constraint(
        "memories_source_message_id_fkey",
        "memories",
        schema="elysia",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "memories_source_message_id_fkey",
        "memories",
        "messages",
        ["source_message_id"],
        ["id"],
        source_schema="elysia",
        referent_schema="elysia",
        ondelete="CASCADE",
    )

    op.create_table(
        "media_assets",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("companion_id", sa.UUID(), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum("audio", "image", name="media_kind", native_enum=False),
            nullable=False,
        ),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("idempotency_key", sa.UUID(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["elysia.users.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["companion_id"], ["elysia.companions.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["elysia.conversations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
        sa.UniqueConstraint(
            "user_id", "kind", "idempotency_key", name="uq_media_user_kind_idempotency"
        ),
        schema="elysia",
    )
    op.create_index(
        "ix_media_assets_user_id", "media_assets", ["user_id"], schema="elysia"
    )
    op.create_index(
        "ix_media_assets_conversation_id",
        "media_assets",
        ["conversation_id"],
        schema="elysia",
    )
    op.create_index(
        "ix_media_assets_companion_id",
        "media_assets",
        ["companion_id"],
        schema="elysia",
    )
    op.create_index(
        "ix_media_assets_created_at", "media_assets", ["created_at"], schema="elysia"
    )

    op.add_column(
        "messages",
        sa.Column(
            "message_type",
            sa.Enum("text", "audio", "image", name="message_type", native_enum=False),
            server_default="text",
            nullable=False,
        ),
        schema="elysia",
    )
    op.add_column(
        "messages", sa.Column("media_asset_id", sa.UUID(), nullable=True), schema="elysia"
    )
    op.add_column(
        "messages", sa.Column("client_request_id", sa.UUID(), nullable=True), schema="elysia"
    )
    op.create_unique_constraint(
        "uq_message_conversation_request",
        "messages",
        ["conversation_id", "client_request_id"],
        schema="elysia",
    )
    op.create_foreign_key(
        "fk_messages_media_asset_id",
        "messages",
        "media_assets",
        ["media_asset_id"],
        ["id"],
        source_schema="elysia",
        referent_schema="elysia",
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_messages_media_asset_id", "messages", ["media_asset_id"], schema="elysia"
    )


def downgrade() -> None:
    op.drop_index("ix_messages_media_asset_id", table_name="messages", schema="elysia")
    op.drop_constraint(
        "fk_messages_media_asset_id", "messages", schema="elysia", type_="foreignkey"
    )
    op.drop_constraint(
        "uq_message_conversation_request", "messages", schema="elysia", type_="unique"
    )
    op.drop_column("messages", "client_request_id", schema="elysia")
    op.drop_column("messages", "media_asset_id", schema="elysia")
    op.drop_column("messages", "message_type", schema="elysia")

    op.drop_index("ix_media_assets_created_at", table_name="media_assets", schema="elysia")
    op.drop_index("ix_media_assets_companion_id", table_name="media_assets", schema="elysia")
    op.drop_index("ix_media_assets_conversation_id", table_name="media_assets", schema="elysia")
    op.drop_index("ix_media_assets_user_id", table_name="media_assets", schema="elysia")
    op.drop_table("media_assets", schema="elysia")
    op.drop_constraint(
        "memories_source_message_id_fkey",
        "memories",
        schema="elysia",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "memories_source_message_id_fkey",
        "memories",
        "messages",
        ["source_message_id"],
        ["id"],
        source_schema="elysia",
        referent_schema="elysia",
        ondelete="SET NULL",
    )
    op.drop_constraint(
        "uq_memory_user_companion_key",
        "memories",
        schema="elysia",
        type_="unique",
    )
    op.drop_column("companions", "voice_config", schema="elysia")
