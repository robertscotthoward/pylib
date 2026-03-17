"""
All data access to anything "persistent" should be done through this module so we have
a single point of entry. Only this module (tables.py) should know which tables exist
and which data stores they are defined in - the main code should not care.

Architecture:
    The persistence layer is configured in config.yaml under the 'persistence' key:

    persistence:
        files:                          # SQLite database for file storage
            source: "sqlite"
            db_path: "cache/files.db"
            tables: INCLUDEFILE(configs/tables_files.yaml)
        kv:                             # SQLite database for key-value storage
            source: "sqlite"
            db_path: "cache/kv.db"
            tables:
                kv: {...}               # Key-value table definition

    Each persistence entry specifies:
    - source: The data source type ("notion" or "sqlite")
    - db_path: (SQLite only) Path to the SQLite database file
    - tables: Table definitions (can be inline or included from external files)

Example:
    SELECT * FROM people JOIN roles ON people.roles = roles.id

Where 'people' is in Notion and 'files' is in SQLite.

Usage:
    from src.models.persistence.tables import get_db, get_db_kv

    db = get_db()
    results = db.query("SELECT * FROM people")

Requirements:
- Every Notion table must have an 'id' field (populated from Notion page ID)
- Tables are configured in config.yaml under persistence.*.tables
"""

from abc import ABC, abstractmethod
from box import BoxList
from cachetools import TTLCache, cached
from diskcache import Cache
from functools import lru_cache
from lib.configurations import *
from lib.tools import *
from sqlite3 import IntegrityError
from typing import Any, Dict, Iterator, List, Optional, Tuple, Set
import os
import sqlite3
import sys
import yaml




class Database(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.connection = None
        self.name = config['name']

    @abstractmethod
    def create_connection(self):
        pass

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        pass

    @abstractmethod
    def query(self, query: str, params: tuple = ()):
        pass

    @abstractmethod
    def get_schema(self) -> List[Dict[str, Any]]:
        """Get the schema of the database."""
        pass

    @abstractmethod
    def ensure_database(self):
        """Ensure database file exists and the tables are created."""
        pass

    @abstractmethod
    def create_schema_diff(self, tables: Box, force: bool = False):
        """Create a schema diff SQL of the database and the tables defined in the config."""
        pass

    def get_connection(self):
        if self.connection is None:
            self.connection = self.create_connection()
        return self.connection




class SqliteDatabase(Database):
    def __init__(self, config: Box):
        super().__init__(config)
        self.path = self.config.get('path')
        self.table_schema = {}
        assert config['type'] == 'sqlite'
        assert config['path']


        
    def create_connection(self):
        if not self.path:
            raise ValueError("Path not configured in config.")
        self.path = os.path.abspath(self.path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Database file not found at {self.path}")
        return sqlite3.connect(self.path)


    def execute(self, sql: str, params: tuple = ()):
        "Execute one or more SQL statements separated by semicolons."
        conn = self.get_connection()
        import re
        last_row_id = -1
        for s in re.split(r';\s*$', sql, flags=re.MULTILINE):
            s = s.strip()
            if s:
                cursor = conn.execute(s, params)
                last_row_id = cursor.lastrowid
                conn.commit()
        return last_row_id


    def query(self, query: str, params: tuple = ()):
        conn = self.get_connection()
        cursor = conn.execute(query, params)
        columns = [
            Box(
                {
                    'cid': cid,
                    'name': column[0],
                    'type': column[1],
                    'display_size': column[2],
                    'internal_size': column[3],
                    'precision': column[4],
                    'scale': column[5],
                    'nullable': column[6]
                }
            )
            for cid, column in enumerate(cursor.description)
        ]
        def boxit(row):
            r = Box({column[0]: row[i] for i, column in enumerate(cursor.description)})
            return r
        rows = cursor.fetchall()
        rows = [boxit(row) for row in rows]
        results = Box(rows=rows, columns=columns)
        return results


    def get_schema(self) -> List[Dict[str, Any]]:
        """Get the schema of the database."""
        conn = self.get_connection()
        tt = {}
        results = self.query(f"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        for row in results.rows:
            tn = row.name
            results = self.get_table_schema(tn)
            tt[tn] = results
        return tt


    def get_table_schema(self, table_name: str) -> BoxList:
        if table_name not in self.table_schema:
            results = self.query(f"PRAGMA table_info({table_name})")
            self.table_schema[table_name] = BoxList(results.columns)
        return self.table_schema[table_name]


    def ensure_database(self, tables: dict):
        """Ensure database file exists and the tables are created."""
        import sqlite3
        ensurePath(self.path)
        if not os.path.exists(self.path):
            # create a new sqlite database at the path
            conn = sqlite3.connect(self.path)
            conn.close()

        ddl = self.create_schema_diff(tables)
        if ddl:
            self.execute(ddl)


    def get_column_expressions(self, column: Dict[str, Any]) -> str:
        """Get the expression for a column."""
        s = f'"{column.name}" {column.type}'
        if column.get('is_primary_key', False):
            s += ' PRIMARY KEY'
            if column.get('autoincrement'):
                s += ' AUTOINCREMENT'
        else:
            if column.get('unique', False):
                s += ' UNIQUE'
            if column.get('notnull', False):
                s += ' NOT NULL'
            if column.get('default'):
                s += f' DEFAULT {column.default}'
        return s

    def create_schema_diff(self, tables: dict, force: bool = False):
        """Create a schema diff SQL of the database and the tables defined in the config."""

        sql = ""
        schema = self.get_schema()
        for table_name, table_config in tables.items():
            if force or table_name not in schema:
                columns = []
                for column in table_config['columns']:
                    colexp = self.get_column_expressions(column)
                    columns.append(f'\n  {colexp}')
                sql += f"CREATE TABLE {table_name} ({', '.join(columns)}\n);\n"
            else:
                for column in table_config['columns']:
                    if force or not any(col.name == column.name for col in schema[table_name]):
                        colexp = self.get_column_expressions(column)
                        sql += f"ALTER TABLE {table_name} ADD COLUMN {colexp};\n"
                    else:
                        tc = next(c for c in schema[table_name] if c.name == column.name)
                        if column.type.lower() != tc.type.lower():
                            sql += f"ALTER TABLE {table_name} ALTER COLUMN {column.name} TYPE {column.type};\n"
                            tc.type = column.type
            sql += "\n"
        return sql

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the SQLite database."""
        if not os.path.exists(self.path):
            return False

        try:
            conn = sqlite3.connect(self.path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception:
            return False


    def get_table_schema(self, table_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get the schema of a SQLite table."""
        if not os.path.exists(self.path):
            return None
        try:
            conn = sqlite3.connect(self.path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            conn.close()

            if not columns:
                return None

            return [
                Box(
                    {
                        'cid': col[0],
                        'name': col[1],
                        'type': col[2],
                        'notnull': col[3],
                        'default': col[4],
                        'pk': col[5],
                    }
                )
                for col in columns
            ]
        except sqlite3.OperationalError:
            return None





    def _insert_sqlite(self, table: str, table_config: dict, data: Dict[str, Any]):
        """Insert a row into a SQLite table using parameterized queries.

        The db_path is already stored in the table_config when tables are registered.
        Excludes autoincrement columns from the INSERT statement.
        If a unique constraint is violated, falls back to upsert.
        """
        import sqlite3

        # Get db_path from the table config (set during registration)
        db_path = table_config.get('db_path')
        if not db_path:
            raise ValueError(f"No database path configured for SQLite table '{table}'")

        # Get column configurations to identify autoincrement columns
        columns_config = {col['name']: col for col in table_config.get('columns', [])}
        
        # Filter out autoincrement columns from data
        filtered_data = {
            col: val for col, val in data.items()
            if not columns_config.get(col, {}).get('autoincrement', False)
        }

        # Use direct sqlite3 connection for INSERT (Shillelagh is read-only for some adapters)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        columns = ', '.join(filtered_data.keys())
        placeholders = ', '.join(['?' for _ in filtered_data.values()])
        values = tuple(filtered_data.values())
        
        try:
            cursor.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                values
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            raise
        finally:
            conn.close()


    def _upsert_sqlite(self, table: str, table_config: dict, data: Dict[str, Any]):
        """Upsert a row into a SQLite table using parameterized queries.

        The db_path is already stored in the table_config when tables are registered.
        Respects the 'update' property on columns - if update is False and the column
        wasn't provided in data, uses the column's default value (e.g., CURRENT_TIMESTAMP).
        Excludes autoincrement columns from the INSERT statement.
        Uses ON CONFLICT with unique columns to handle duplicates.
        Automatically updates UpdatedAt to CURRENT_TIMESTAMP on conflict.
        """
        import sqlite3

        # Get db_path from the table config (set during registration)
        db_path = table_config.get('db_path')
        if not db_path:
            raise ValueError(f"No database path configured for SQLite table '{table}'")

        # Use direct sqlite3 connection for INSERT (Shillelagh is read-only for some adapters)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get column configurations
        columns_config = {col['name']: col for col in table_config.get('columns', [])}
        
        # Filter out autoincrement columns from data for INSERT
        filtered_data = {
            col: val for col, val in data.items()
            if not columns_config.get(col, {}).get('autoincrement', False)
        }
        
        # Find unique columns for ON CONFLICT clause
        unique_columns = [col_name for col_name, col_config in columns_config.items() 
                         if col_config.get('unique', False)]
        
        if not unique_columns:
            raise ValueError(f"Table '{table}' has no unique columns defined for upsert")
        
        # Build the SET clause for updating existing rows
        # Only include columns that should be updated (update != False)
        set_parts = []
        
        for col_name, col_config in columns_config.items():
            if col_name.lower() == 'id':
                continue
            
            # Skip autoincrement columns in updates
            if col_config.get('autoincrement', False):
                continue
            
            # Skip unique columns (they shouldn't be updated)
            if col_config.get('unique', False):
                continue
            
            # Check if column should be updated
            should_update = col_config.get('update', True)
            
            if col_name in data and should_update:
                # Column was provided in data and should be updated, use excluded value
                set_parts.append(f"{col_name} = excluded.{col_name}")
            elif not should_update and col_config.get('default'):
                # Column has update=False and has a default, use the default
                default = col_config.get('default')
                set_parts.append(f"{col_name} = {default}")
        
        # Always update UpdatedAt to CURRENT_TIMESTAMP on conflict (if column exists)
        if 'UpdatedAt' in columns_config:
            # Remove any existing UpdatedAt from set_parts to avoid duplicates
            set_parts = [p for p in set_parts if not p.startswith('UpdatedAt')]
            set_parts.append("UpdatedAt = CURRENT_TIMESTAMP")
        
        set_clause = ', '.join(set_parts)
        if set_clause:
            set_clause = f"SET {set_clause}"
        
        # Build the ON CONFLICT clause
        conflict_col_list = ', '.join(unique_columns)
        on_conflict = f"ON CONFLICT({conflict_col_list}) DO UPDATE {set_clause}" if set_clause else ""
        
        # Build the INSERT INTO statement
        insert_sql = f"INSERT INTO {table} ({', '.join(filtered_data.keys())}) VALUES ({', '.join(['?' for _ in filtered_data.values()])}) {on_conflict}"
        
        # Use only the filtered_data values for parameterized query (no set_values needed with excluded)
        all_values = tuple(filtered_data.values())
        cursor.execute(insert_sql, all_values)
        conn.commit()
        conn.close()




# ============================================================================
# Testing Functions
# ============================================================================


def init_tests():
    yamlPath = "tests/data/configs/config1.yaml"
    config, credentials, environment = get_config_credentials_environment(yamlPath, None)
    settings = Box(config)
    database = SqliteDatabase(settings.databases.main)
    return settings, database


def ensure_database():
    settings, database = init_tests()
    database.ensure_database(settings.tables)


def test_read_one_table():
    """Test reading one table from the database."""
    db = get_db()
    sql = "select * from roles"
    rows = db.query(sql)
    print(rows)


def test_read_one_table_direct():
    """Test reading one table from the database."""

    # Get the notion token
    token = settings['notion.cei.token']

    # Get the roles table from the database


def test1():
    """Test the unified adapter with JOINs across Notion tables.

    Uses Shillelagh for Notion table access (people, roles, privileges are Notion tables).
    """
    db = get_db()
    results = db.query("""
        SELECT *
        FROM people
        JOIN roles ON people.roles = roles.id
    """)
    print(results)

    # Use Shillelagh connection for Notion table queries
    conn = get_shillelagh_connection()

    # Test 1: Simple SELECT from each table
    print("=" * 60)
    print("TEST 1: Simple SELECT queries")
    print("=" * 60)

    people_result = conn.execute("SELECT first, last FROM people")
    people_rows = people_result.fetchall()
    print(f"People: {len(people_rows)} rows")
    assert len(people_rows) >= 1, "Should have at least 1 person"

    roles_result = conn.execute("SELECT id, name FROM roles")
    roles_rows = roles_result.fetchall()
    print(f"Roles: {len(roles_rows)} rows")
    assert len(roles_rows) >= 1, "Should have at least 1 role"

    privileges_result = conn.execute("SELECT id, name FROM privileges")
    privileges_rows = privileges_result.fetchall()
    print(f"Privileges: {len(privileges_rows)} rows")
    assert len(privileges_rows) >= 1, "Should have at least 1 privilege"

    # Test 2: Two-table JOIN (people -> roles)
    print()
    print("=" * 60)
    print("TEST 2: Two-table JOIN (people -> roles)")
    print("=" * 60)

    cursor = conn.execute("""
        SELECT *
        FROM people
        JOIN roles ON people.roles = roles.id
    """)
    column_names = [column[0] for column in cursor.description]
    join_rows = cursor.fetchall()
    print(f"People with roles: {len(join_rows)} rows")
    for row in join_rows:
        print(f"  {row}")
    assert len(join_rows) >= 1, "Should have at least 1 person with a role"


    # Test 3: Two-table JOIN (roles -> privileges)
    print()
    print("=" * 60)
    print("TEST 3: Two-table JOIN (roles -> privileges)")
    print("=" * 60)

    roles_privs_result = conn.execute("""
        SELECT roles.name as role_name, privileges.name as privilege_name
        FROM roles
        JOIN privileges ON roles.privileges = privileges.id
    """)
    roles_privs_rows = roles_privs_result.fetchall()
    print(f"Roles with privileges: {len(roles_privs_rows)} rows")
    for row in roles_privs_rows:
        print(f"  {row}")
    assert len(roles_privs_rows) >= 1, "Should have at least 1 role with privileges"

    # Test 4: Three-table JOIN (people -> roles -> privileges)
    print()
    print("=" * 60)
    print("TEST 4: Three-table JOIN (people -> roles -> privileges)")
    print("=" * 60)

    full_join_result = conn.execute("""
        SELECT
            people.first,
            people.last,
            roles.name as role_name,
            privileges.name as privilege_name
        FROM people
        JOIN roles ON people.roles = roles.id
        JOIN privileges ON roles.privileges = privileges.id
    """)
    full_join_rows = full_join_result.fetchall()
    print(f"People with roles and privileges: {len(full_join_rows)} rows")
    for row in full_join_rows:
        print(f"  {row}")
    assert len(full_join_rows) >= 1, "Should have at least 1 person with roles and privileges"
    # Note: This may be 0 if no person has a role with privileges linked
    # The JOIN logic is verified by Test 2 (people->roles) and Test 3 (roles->privileges)

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


def test2():
    """Test SQLite connection with attached databases."""
    conn = get_connection()
   

def inspect_all_tables():
    """Inspect all tables defined in tables.yaml and print their columns and types."""
    import requests
    
    print("=" * 80)
    print("NOTION TABLES SCHEMA INSPECTION")
    print("=" * 80)
    
    # Get all tables from config
    tables_config = settings['persistence.tables'] or {}
    
    for table_name, table_config in tables_config.items():
        print(f"\n{'-' * 80}")
        print(f"TABLE: {table_name}")
        print(f"{'-' * 80}")
        
        if table_config.get("source") != "notion":
            print(f"  Source: {table_config.get('source')} (not Notion)")
            continue
        
        notion_id = table_config.get("notionid")
        if not notion_id:
            print("  ERROR: No Notion ID configured")
            continue
        
        # Format the database ID
        db_id = notion_id.replace("-", "")
        
        # Get database schema from Notion
        url = f"https://api.notion.com/v1/databases/{db_id}"
        
        headers = {
            "Authorization": f"Bearer {settings['notion.cei.token']}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                properties = data.get("properties", {})
                
                print(f"  Notion ID: {notion_id}")
                print(f"  Total Columns: {len(properties)}")
                print(f"\n  Columns:")
                
                for prop_name, prop_config in properties.items():
                    prop_type = prop_config.get("type", "unknown")
                    print(f"    - {prop_name:<30} : {prop_type}")
                    
                    # Print additional details for complex types
                    if prop_type == "select":
                        options = prop_config.get("select", {}).get("options", [])
                        if options:
                            print(f"      Options: {', '.join([o.get('name', '') for o in options])}")
                    elif prop_type == "multi_select":
                        options = prop_config.get("multi_select", {}).get("options", [])
                        if options:
                            print(f"      Options: {', '.join([o.get('name', '') for o in options])}")
                    elif prop_type == "relation":
                        relation_db = prop_config.get("relation", {}).get("database_id", "")
                        print(f"      Related to: {relation_db}")
                    elif prop_type == "formula":
                        formula = prop_config.get("formula", {}).get("expression", "")
                        print(f"      Formula: {formula}")
                    elif prop_type == "rollup":
                        rollup_prop = prop_config.get("rollup", {}).get("relation_property_name", "")
                        rollup_func = prop_config.get("rollup", {}).get("function", "")
                        print(f"      Rollup: {rollup_prop} -> {rollup_func}")
            else:
                print(f"  ERROR: Could not fetch schema ({response.status_code})")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'=' * 80}\n")


def test_join_query():
    """Test joining People -> Roles -> Privileges using Shillelagh for Notion tables."""
    conn = get_shillelagh_connection()
    
    print("=" * 80)
    print("PEOPLE WITH RESOLVED RELATIONS")
    print("=" * 80)
    
    # Query people table
    result = conn.execute("SELECT * FROM people")
    rows = result.fetchall()
    
    print(f"\nFound {len(rows)} people:\n")
    
    for row in rows:
        print(f"  {row['first']} {row['last']}")
        if 'roles' in row and row['roles']:
            print(f"    Roles: {row['roles']}")
    
    # Query roles with privileges
    print(f"\n{'=' * 80}")
    print("ROLES WITH RESOLVED PRIVILEGES")
    print(f"{'=' * 80}\n")
    
    roles_result = conn.execute("SELECT * FROM roles")
    roles_rows = roles_result.fetchall()
    
    print(f"Found {len(roles_rows)} roles:\n")
    
    for role in roles_rows:
        print(f"  {role['name']}")
        if 'privileges' in role and role['privileges']:
            print(f"    Privileges: {role['privileges']}")
    
    print(f"\n{'=' * 80}\n")
    conn.close()


def dump_all_notion_tables():
    """Dump all columns, types, and records for each Notion table."""
    import requests
    
    print("=" * 100)
    print("COMPLETE NOTION DATABASE DUMP")
    print("=" * 100)
    
    # Get all tables from config
    tables_config = settings['persistence.tables'] or {}
    
    for table_name, table_config in tables_config.items():
        if table_config.get("source") != "notion":
            continue
        
        notion_id = table_config.get("notionid")
        if not notion_id:
            continue
        
        db_id = notion_id.replace("-", "")
        url = f"https://api.notion.com/v1/databases/{db_id}/query"
        
        headers = {
            "Authorization": f"Bearer {settings['notion.cei.token']}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        
        print(f"\n{'=' * 100}")
        print(f"TABLE: {table_name.upper()}")
        print(f"{'=' * 100}")
        
        response = requests.post(url, headers=headers, json={})
        
        if response.status_code == 200:
            data = response.json()
            
            # Get schema from first result
            if data.get("results"):
                first_page = data["results"][0]
                properties = first_page.get("properties", {})
                
                print(f"\nCOLUMNS AND TYPES ({len(properties)} total):")
                print("-" * 100)
                for prop_name, prop_config in properties.items():
                    prop_type = prop_config.get("type", "unknown")
                    print(f"  {prop_name:<30} : {prop_type}")
                
                # Print all records
                print(f"\nRECORDS ({len(data.get('results', []))} total):")
                print("-" * 100)
                
                for idx, page in enumerate(data.get("results", []), 1):
                    print(f"\n  Record {idx}:")
                    properties = page.get("properties", {})
                    
                    for prop_name, prop_value in properties.items():
                        prop_type = prop_value.get("type", "unknown")
                        
                        # Extract value based on type
                        if prop_type == "title":
                            value = "".join([t.get("plain_text", "") for t in prop_value.get("title", [])])
                        elif prop_type == "rich_text":
                            value = "".join([t.get("plain_text", "") for t in prop_value.get("rich_text", [])])
                        elif prop_type == "select":
                            value = prop_value.get("select", {}).get("name", "")
                        elif prop_type == "multi_select":
                            value = ", ".join([s.get("name", "") for s in prop_value.get("multi_select", [])])
                        elif prop_type == "number":
                            value = prop_value.get("number")
                        elif prop_type == "checkbox":
                            value = prop_value.get("checkbox")
                        elif prop_type == "email":
                            value = prop_value.get("email", "")
                        elif prop_type == "phone_number":
                            value = prop_value.get("phone_number", "")
                        elif prop_type == "url":
                            value = prop_value.get("url", "")
                        elif prop_type == "relation":
                            relation_ids = [r.get("id") for r in prop_value.get("relation", [])]
                            value = f"[{len(relation_ids)} relations: {', '.join(relation_ids[:3])}{'...' if len(relation_ids) > 3 else ''}]"
                        elif prop_type == "unique_id":
                            unique_id = prop_value.get("unique_id", {})
                            value = f"{unique_id.get('prefix', '')}{unique_id.get('number', '')}"
                        else:
                            value = str(prop_value.get(prop_type, ""))
                        
                        print(f"    {prop_name:<28} ({prop_type:<15}): {value}")
        else:
            print(f"  ERROR: {response.status_code} - {response.text}")
    
    print(f"\n{'=' * 100}\n")


def create_sqlite_table(table_name: str, db_path: str, tables_config: dict) -> bool:
    """
    Create a SQLite table based on the schema defined in config.

    Args:
        table_name: The table name key from tables config (e.g., 'files', 'kv')
        db_path: Path to SQLite database file
        tables_config: The tables configuration dictionary

    Returns:
        True if table was created or already exists, False if table_name not found in config

    Raises:
        Exception: If SQLite creation fails
    """
    import sqlite3

    db_path = os.path.abspath(db_path)

    # Get table config
    if table_name not in tables_config:
        print(f"[ERROR] Table '{table_name}' not found in tables config")
        return False

    table_config = tables_config[table_name]

    # Map column types to SQLite types
    # Includes both Notion-style types and native SQLite types
    type_mapping = {
        # Notion property types
        'title': 'TEXT NOT NULL',
        'rich_text': 'TEXT',
        'email': 'TEXT',
        'phone_number': 'TEXT',
        'url': 'TEXT',
        'number': 'REAL',
        'checkbox': 'BOOLEAN',
        'date': 'TEXT',
        'select': 'TEXT',
        'multi_select': 'TEXT',
        'relation': 'TEXT',
        'unique_id': 'TEXT',
        'place': 'TEXT',
        # Native SQLite types
        'text': 'TEXT',
        'integer': 'INTEGER',
        'real': 'REAL',
        'blob': 'BLOB',
    }

    # Build CREATE TABLE statement
    columns = table_config.get('columns', [])
    if not columns:
        print(f"[WARN] No columns defined for table '{table_name}' in config")
        return False

    column_defs = []
    for col in columns:
        col_name = col.get('name', '').replace(' ', '_')
        col_type = col.get('type', 'TEXT').lower()
        sql_type = type_mapping.get(col_type, 'TEXT')

        # Handle primary key
        if col.get('is_primary_key'):
            sql_type += ' PRIMARY KEY'
            # Add AUTOINCREMENT if specified
            if col.get('autoincrement'):
                sql_type += ' AUTOINCREMENT'
        
        # Handle unique constraint
        if col.get('unique', False):
            sql_type += ' UNIQUE'

        column_defs.append(f"  {col_name} {sql_type}")

    # Check if there's already a primary key defined
    has_primary_key = any(col.get('is_primary_key') for col in columns)

    # Only add 'id' column if no primary key is defined
    if not has_primary_key:
        has_id = any(col.get('name') == 'ID' or col.get('name') == 'id' for col in columns)
        if not has_id:
            column_defs.insert(0, "  id TEXT PRIMARY KEY")
    
    create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
{','.join(column_defs)}
);
"""
    
    try:
        conn = sqlite3.connect(os.path.abspath(db_path))
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        conn.close()
        print(f"[OK] SQLite table '{table_name}' created/verified successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create SQLite table '{table_name}': {str(e)}")
        raise


def create_all_sqlite_tables():
    """
    Create all SQLite tables defined in the persistence config.

    Iterates through all persistence configurations and creates
    SQLite tables for persistence entries with source='sqlite'.
    """
    for k, persistence in settings['persistence'].items():
        pconfig = _get_persistence_config(persistence)
        source = pconfig.get('source', 'sqlite')

        # Skip non-SQLite sources (like Notion)
        if source != 'sqlite':
            continue

        db_path = pconfig.get('db_path')
        if not db_path:
            print(f"[WARN] No db_path for persistence.{k}, skipping")
            continue

        tables_config = settings[f'persistence.{k}.tables']
        if tables_config is None:
            continue

        # Convert Settings object to dict properly
        if hasattr(tables_config, 'to_dict'):
            tables_dict = tables_config.to_dict()
        elif hasattr(tables_config, 'config'):
            tables_dict = tables_config.config
        elif isinstance(tables_config, dict):
            tables_dict = tables_config
        else:
            # Try iterating through items
            tables_dict = {name: (tbl.to_dict() if hasattr(tbl, 'to_dict') else tbl)
                          for name, tbl in tables_config.items()}

        results = {}

        print(f"\n{'='*60}")
        print(f"Creating SQLite tables for persistence.{k}")
        print(f"Database: {db_path}")
        print(f"{'='*60}\n")

        for table_name, table in tables_dict.items():
            try:
                # Convert table config if needed
                if hasattr(table, 'to_dict'):
                    table = table.to_dict()
                elif hasattr(table, 'config'):
                    table = table.config

                success = create_sqlite_table(table_name, db_path, tables_dict)
                results[table_name] = success
            except Exception as e:
                print(f"[ERROR] Exception creating table '{table_name}': {str(e)}")
                results[table_name] = False

        print(f"\n{'='*60}")
        passed = sum(1 for v in results.values() if v)
        failed = len(results) - passed
        print(f"Results: {passed} passed, {failed} failed")
        print(f"{'='*60}\n")
    



_cached_db = {}
def get_db(path = None) -> Database:
    if path is None:
        path = settings['persistence.files.db_path']
    if path not in _cached_db:
        _cached_db[path] = Database(path)
    return _cached_db[path]

def get_db_kv(path = None) -> Database:
    """Get a Database instance for the key-value store."""
    if path is None:
        path = settings['persistence.kv.db_path']

    if path not in _cached_db:
        _cached_db[path] = Database(path)
    return _cached_db[path]


def kv_get(key: str) -> Optional[str]:
    """Get a value from the key-value store."""
    import sqlite3

    db_path = os.path.abspath(settings['persistence.kv.db_path'])
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT Value FROM kv WHERE ID = ?", (key,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]
    return None


def kv_set(key: str, value: str):
    """Set a value in the key-value store (insert or update)."""
    import sqlite3

    db_path = os.path.abspath(settings['persistence.kv.db_path'])
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO kv (ID, Value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()
    conn.close()


def kv_delete(key: str):
    """Delete a value from the key-value store."""
    import sqlite3

    db_path = os.path.abspath(settings['persistence.kv.db_path'])
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM kv WHERE ID = ?", (key,))
    conn.commit()
    conn.close()


def test_kv():
    """Test the key-value store."""
    print("=" * 60)
    print("TEST: Key-Value Store")
    print("=" * 60)

    # Ensure tables exist
    create_all_sqlite_tables()

    # Test set
    kv_set("test_key", "test_value")
    print("[OK] kv_set('test_key', 'test_value')")

    # Test get
    value = kv_get("test_key")
    assert value == "test_value", f"Expected 'test_value', got '{value}'"
    print(f"[OK] kv_get('test_key') = '{value}'")

    # Test update
    kv_set("test_key", "updated_value")
    value = kv_get("test_key")
    assert value == "updated_value", f"Expected 'updated_value', got '{value}'"
    print(f"[OK] kv_set update: kv_get('test_key') = '{value}'")

    # Test delete
    kv_delete("test_key")
    value = kv_get("test_key")
    assert value is None, f"Expected None after delete, got '{value}'"
    print("[OK] kv_delete('test_key') - value is None")

    print("=" * 60)
    print("ALL KV TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.append("ensure-database")

    with Spy():
        if len(sys.argv) > 1:
            if sys.argv[1] == "dump":
                dump_all_notion_tables()
            elif sys.argv[1] == "ensure-database":
                ensure_database()
            elif sys.argv[1] == "join":
                test_join_query()
            elif sys.argv[1] == "test_read_one_table":
                test_read_one_table()
            elif sys.argv[1] == "inspect":
                inspect_all_tables()
            elif sys.argv[1] == "create-tables":
                # Create all SQLite tables from all persistence configs
                create_all_sqlite_tables()
            elif sys.argv[1] == "test-kv":
                # Test the key-value store
                test_kv()
            elif sys.argv[1] == "test":
                # Run basic test
                test1()
            elif sys.argv[1] == "test-all":
                # Run comprehensive tests
                create_all_sqlite_tables()
                test1()
                test_kv()
            else:
                # Default: create tables
                create_all_sqlite_tables()
                test_read_one_table()
        elif len(sys.argv) > 1 and sys.argv[1] == "join":
            test_join_query()
        elif len(sys.argv) > 1 and sys.argv[1] == "test_read_one_table":
            test_read_one_table()
        elif len(sys.argv) > 1 and sys.argv[1] == "inspect":
            inspect_all_tables()
        elif len(sys.argv) > 1 and sys.argv[1] == "create-tables":
            # Create all SQLite tables from all persistence configs
            create_all_sqlite_tables()
        elif len(sys.argv) > 1 and sys.argv[1] == "test-kv":
            # Test the key-value store
            test_kv()
        elif len(sys.argv) > 1 and sys.argv[1] == "test":
            # Run basic test
            test1()
        elif len(sys.argv) > 1 and sys.argv[1] == "test-all":
            # Run comprehensive tests
            create_all_sqlite_tables()
            test1()
            test_kv()
        else:
            # Default: create tables
            # create_all_sqlite_tables()
            test_read_one_table()
