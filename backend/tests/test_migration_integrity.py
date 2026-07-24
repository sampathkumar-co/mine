from __future__ import annotations

import os
import subprocess
from pathlib import Path

from sqlalchemy import create_engine, inspect

from app.core.database import Base, import_models


def test_migrations_are_immutable_and_match_model_schema(tmp_path: Path) -> None:
    migration_dir = Path(__file__).parents[1] / "alembic" / "versions"
    for path in migration_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import_models" not in source
        assert "Base.metadata" not in source

    database = tmp_path / "migration-integrity.db"
    url = f"sqlite+pysqlite:///{database}"
    env = {
        **os.environ,
        "DIRECTOR_ENVIRONMENT": "test",
        "DIRECTOR_DATABASE_URL": url,
        "DIRECTOR_AUTH_SECRET": "migration-integrity-secret-change-me",
    }
    subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=Path(__file__).parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    import_models()
    actual = inspect(create_engine(url))
    actual_tables = set(actual.get_table_names()) - {"alembic_version"}
    expected_tables = set(Base.metadata.tables)
    assert actual_tables == expected_tables
    for table_name in expected_tables:
        actual_columns = {column["name"] for column in actual.get_columns(table_name)}
        expected_columns = {column.name for column in Base.metadata.tables[table_name].columns}
        assert actual_columns == expected_columns, table_name
