"""Database migration utilities for CEI OS.

This module handles schema comparison and migration generation for both
SQLite and Notion data sources.

Commands:
    migrate - Generate migration SQL files without applying them
    update  - Create databases or apply migrations
"""

from abc import ABC, abstractmethod
import os
import re
import sqlite3
from box import Box
import requests
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from lib.database.databases import Database, SqliteDatabase
from lib.tools import *
from datetime import datetime


def default_log(message: str) -> None:
    print(message)

def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    return ' '.join(sql.split()).upper()


class Migrations(ABC):
    def __init__(self, settings: Box, database: Database, log: Callable = default_log):
        assert settings.tables
        self.settings = settings
        self.database = database
        self.name = database.name
        self.path = database.path
        self.migrations_path = Path(self.settings.persistence.migrations_path)
        self.log = log

    def migrate(self) -> Dict[str, Optional[str]]:
        """
        Generate migration SQL files for all SQLite persistence sources.
        Save this SQL to the migrations_path.

        This command:
        - Does NOT modify any existing database files
        - For SQLite: Generates CREATE/ALTER statements
        - Writes to data/migrations/DDDD-NNNN.sql
        - Always overwrites DDDD-0000.sql with complete creation scripts

        Returns:
            Dict mapping database names to migration file paths (or None if no changes)
        """
        self.log("[*] Generating migration files...")
        self.log("")

        ensurePath(str(self.migrations_path))

        results = {}

        self.log(f"[*] Processing SQLite source: {self.name}")

        # If database doesn't exist, skip (will be created by update command)
        if not os.path.exists(self.path):
            self.log(f"    [INFO] Database file does not exist: {self.path}")
            self.log(f"    [INFO] Run 'update' command to create it")
            results[self.name] = None

            # Still generate the base file (0000) with full creation scripts
            self._generate_base_migration()
            return

        # Generate migration for existing database
        M = []
        M.append(f"-- Migration for '{self.name}'")
        M.append(f"-- Generated at {datetime.datetime.now().isoformat()}")
        M.append("")

        # Get previous migration statements to avoid duplicates
        previous_statements = self._read_previous_migration()

        has_changes = False

        for table_name, table_cfg in self.settings.tables.items():
            self.log(f"    [*] Checking table: {table_name}")

            if not self.database.table_exists(table_name):
                # Table doesn't exist - generate CREATE TABLE
                create_sql = self.database.create_schema_diff(self.settings.tables)
                if create_sql:
                    # Check if this statement is already in previous migration
                    if _normalize_sql(create_sql) not in {_normalize_sql(s) for s in previous_statements}:
                        has_changes = True
                        M.append(f"-- Create table '{table_name}'")
                        M.append(create_sql)
                        M.append("")
            else:
                # Table exists - check for schema differences
                existing_schema = self.database.get_table_schema(table_name)
                if existing_schema:
                    alterations = self.database.create_schema_diff(self.settings.tables)
                    if alterations:
                        has_changes = True
                        M.append(alterations)
                        M.append("")

        if has_changes:
            # Write incremental migration file
            migration_num = self._get_next_migration_number()
            migration_file = self.migrations_path / f"{self.name}-{migration_num}.sql"
            migration_file.write_text('\n'.join(M), encoding='utf-8')
            self.log(f"    [OK] Migration file created: {migration_file}")
            results[self.name] = str(migration_file)
        else:
            self.log(f"    [OK] No schema changes detected")
            results[self.name] = None

        # Always regenerate the base file (0000) with complete creation scripts
        self._generate_base_migration()

        self.log("")
        self.log("[OK] Migration generation complete")

        return results


    def _generate_base_migration(self):
        """Generate the base migration file (DDDD-0000.sql) with complete creation scripts."""
        M = []
        M.append(f"-- Base schema for '{self.name}'")
        M.append(f"-- Generated at {datetime.datetime.now().isoformat()}")
        M.append(f"-- This file contains complete CREATE TABLE statements for all tables")
        M.append("")

        tables = self.database.get_schema()
        create_sql = self.database.create_schema_diff(self.settings.tables, force=True)
        base_file = Path(self.migrations_path) / f"{self.name}-0000.sql"
        base_file.write_text(create_sql, encoding='utf-8')
        self.log(f"    [OK] Base schema file updated: {base_file}")





    def _get_next_migration_number(self) -> str:
        """Get the next migration file number for a specific database."""
        migrations_path = Path(self.migrations_path)

        if not migrations_path.exists():
            return "0001"

        # Look for files matching DDDD-NNNN.sql pattern for this database
        pattern = f"{self.name}-*.sql"
        existing_files = list(migrations_path.glob(pattern))

        if not existing_files:
            return "0001"

        # Extract numbers from filenames (format: DDDD-NNNN.sql)
        numbers = []
        for f in existing_files:
            match = re.match(rf"{re.escape(self.name)}-(\d+)\.sql", f.name)
            if match:
                num = int(match.group(1))
                if num > 0:  # Skip 0000 (the base file)
                    numbers.append(num)

        if not numbers:
            return "0001"

        next_num = max(numbers) + 1
        return f"{next_num:04d}"


    def _get_latest_migration_file(self) -> Optional[Path]:
        """Get the most recent migration file for a database."""
        migrations_path = Path(self.migrations_path)

        if not migrations_path.exists():
            return None

        pattern = f"{self.name}-*.sql"
        existing_files = list(migrations_path.glob(pattern))

        if not existing_files:
            return None

        # Find the highest numbered migration (excluding 0000)
        latest = None
        latest_num = 0

        for f in existing_files:
            match = re.match(rf"{re.escape(self.name)}-(\d+)\.sql", f.name)
            if match:
                num = int(match.group(1))
                if num > latest_num:
                    latest_num = num
                    latest = f

        return latest


    def _read_previous_migration(self) -> set:
        """Read statements from the previous migration file."""
        latest = self._get_latest_migration_file()
        if not latest or not latest.exists():
            return set()

        content = latest.read_text(encoding='utf-8')

        # Extract SQL statements (non-comment lines)
        statements = set()
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):
                statements.add(line)

        return statements






class SqliteMigrations(Migrations):
    def __init__(self, settings: Box, database: Database, log: Callable = default_log):
        super().__init__(settings, database, log)
    











# ============================================================================
# Helper Functions - Config Extraction
# ============================================================================

def _get_persistence_config(persistence) -> dict:
    """Extract config values from a persistence Settings object."""
    if hasattr(persistence, 'to_dict'):
        return persistence.to_dict()
    elif hasattr(persistence, 'config'):
        return persistence.config
    elif hasattr(persistence, '__getitem__'):
        result = {}
        for key in ['source', 'db_path', 'tables']:
            try:
                val = persistence[key]
                if hasattr(val, 'to_dict'):
                    result[key] = val.to_dict()
                elif hasattr(val, 'config'):
                    result[key] = val.config
                else:
                    result[key] = val
            except (KeyError, TypeError):
                pass
        return result
    return persistence if isinstance(persistence, dict) else {}


def _get_tables_dict(tables_config) -> dict:
    """Convert tables config to a plain dictionary."""
    if tables_config is None:
        return {}
    if hasattr(tables_config, 'to_dict'):
        return tables_config.to_dict()
    elif hasattr(tables_config, 'config'):
        return tables_config.config
    elif isinstance(tables_config, dict):
        return tables_config
    else:
        result = {}
        for name, tbl in tables_config.items():
            if hasattr(tbl, 'to_dict'):
                result[name] = tbl.to_dict()
            elif hasattr(tbl, 'config'):
                result[name] = tbl.config
            else:
                result[name] = tbl
        return result


    def _get_next_migration_number(self) -> str:
        """Get the next migration file number for a specific database."""
        migrations_path = self.migrations_path
        if not migrations_path.exists():
            return "0001"
        pattern = f"{self.name}-*.sql"
        existing_files = list(migrations_path.glob(pattern))
        if not existing_files:
            return "0001"
        # Extract numbers from filenames (format: DDDD-NNNN.sql)
        numbers = []
        for f in existing_files:
            match = re.match(rf"{re.escape(db_name)}-(\d+)\.sql", f.name)
            if match:
                num = int(match.group(1))
                if num > 0:  # Skip 0000 (the base file)
                    numbers.append(num)

        if not numbers:
            return "0001"

        next_num = max(numbers) + 1
        return f"{next_num:04d}"


    def _get_latest_migration_file(self) -> Optional[Path]:
        """Get the most recent migration file for a database."""
        migrations_path = Path(self.migrations_path)

        if not migrations_path.exists():
            return None

        pattern = f"{self.name}-*.sql"
        existing_files = list(migrations_path.glob(pattern))

        if not existing_files:
            return None

        # Find the highest numbered migration (excluding 0000)
        latest = None
        latest_num = 0

        for f in existing_files:
            match = re.match(rf"{re.escape(self.name)}-(\d+)\.sql", f.name)
            if match:
                num = int(match.group(1))
                if num > latest_num:
                    latest_num = num
                    latest = f

        return latest


    def _read_previous_migration(self) -> set:
        """Read statements from the previous migration file."""
        latest = self._get_latest_migration_file()
        if not latest or not latest.exists():
            return set()

        content = latest.read_text(encoding='utf-8')

        # Extract SQL statements (non-comment lines)
        statements = set()
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):
                statements.add(line)

        return statements


# ============================================================================
# Main Functions - migrate command
# ============================================================================


if __name__ == "__main__":
    from lib.tools import *
    from lib.configurations import *
    yamlPath = "./tests/data/configs/config1.yaml"
    config, credentials, environment = get_config_credentials_environment(yamlPath, None)
    settings = Box(config)
    database = SqliteDatabase(settings.databases.main)
    migrations = Migrations(settings, database)
    migrations.migrate()
