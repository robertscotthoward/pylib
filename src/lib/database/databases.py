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
        ct, comment = get_sqlite_type(column.type)
        if not ct:
            raise ValueError(f"Unknown column type: {column.type}")
        s = f'"{column.name}" {ct}'
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
        return s, comment

    def create_schema_diff(self, tables: dict, force: bool = False):
        """Create a schema diff SQL of the database and the tables defined in the config."""

        sql = ""
        schema = self.get_schema()
        for table_name, table_config in tables.items():
            if force or table_name not in schema:
                columns = []
                for column in table_config['columns']:
                    desc = ''
                    if column.get('description'):
                        desc = f' -- {column.description}'
                    colexp, comment = self.get_column_expressions(column)
                    if column is table_config['columns'][-1]:
                        columns.append(f'\n  {colexp}{comment}{desc}\n')
                    else:
                        columns.append(f'\n  {colexp},{comment}{desc}')
                sql += f"CREATE TABLE {table_name} ({''.join(columns)});\n"
            else:
                for column in table_config['columns']:
                    if force or not any(col.name == column.name for col in schema[table_name]):
                        colexp, comment = self.get_column_expressions(column)
                        sql += f"ALTER TABLE {table_name} ADD COLUMN {colexp}; {comment}\n"
                    else:
                        tc = next(c for c in schema[table_name] if c.name == column.name)
                        if column.type.lower() != tc.type.lower():
                            sql += f"ALTER TABLE {table_name} ALTER COLUMN {column.name} TYPE {column.type}; {comment}\n"
                            tc.type = column.type
            sql += "\n"
        return sql.strip()

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

import sys
from typing import Optional, Type

def find_class_in_modules(class_name: str) -> Optional[Type]:
    """
    Search all loaded modules in the runtime for a class by name.
    
    Returns the first matching class found, or None if not found.
    
    Args:
        class_name: The name of the class to search for
        
    Returns:
        The class object if found, None otherwise
        
    Example:
        MyClass = find_class_in_modules("MyClass")
        if MyClass:
            instance = MyClass()
    """
    import enum
    import inspect
    from src.core.interfaces.types import DataType

    modules = list(sys.modules.values())
    for module in modules:
        # Skip None modules and built-in modules
        if module is None:
            continue
            
        spec = getattr(module, "__spec__", None)
        if not spec:
            continue

        is_builtin = getattr(spec, "origin", None) == "built-in"
        if is_builtin:
            continue

        cls = getattr(module, class_name, None)
        if not cls:
            continue

        if not inspect.isclass(cls):
            continue

        if issubclass(cls, enum.Enum):
            return cls

        if issubclass(cls, DataType):
            return cls
   
    return None


def get_sqlite_type(col_type: str) -> str:
    import enum
    from src.core.interfaces.types import DataType
    if col_type in type_mapping:
        if col_type != type_mapping[col_type]:
            return type_mapping[col_type], f' -- {col_type}'
        return type_mapping[col_type], ''

    cls = find_class_in_modules(col_type)
    if cls:
        if isinstance(cls, enum.Enum):
            return 'TEXT', f' -- {cls.__name__}'
        if issubclass(cls, DataType):
            return cls.sqlite_type, f' -- {cls.__name__}'

    raise ValueError(f"Unknown column type: {col_type}")



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

    columns = table_config.get('columns', [])
    if not columns:
        print(f"[WARN] No columns defined for table '{table_name}' in config")
        return False

    column_defs = []
    for col in columns:
        col_name = col.get('name', '').replace(' ', '_')
        col_type = col.get('type', 'TEXT').lower()
        sql_type = get_sqlite_type(col_type)
        if not sql_type:
            raise ValueError(f"Unknown column type: {col_type}")


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
