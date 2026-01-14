"""Data tools - JSON, CSV, SQL transformations."""

import csv
import io
import json
import sqlite3
from typing import Any


def json_query(data: Any, query: str) -> Any:
    """Query JSON data using JMESPath-like expressions.

    Supports basic path expressions:
    - foo.bar - nested access
    - foo[0] - array index
    - foo[*].bar - map over array
    - foo[?bar > 10] - filter (limited)
    """
    # Simple implementation without jmespath dependency
    parts = query.replace("[", ".").replace("]", "").split(".")
    result = data

    for part in parts:
        if not part:
            continue
        if part == "*":
            # Map over array
            if isinstance(result, list):
                continue  # Will be handled by next part
            else:
                return None
        elif part.startswith("?"):
            # Filter - simplified
            return result  # TODO: implement filtering
        elif part.isdigit():
            # Array index
            idx = int(part)
            if isinstance(result, list) and idx < len(result):
                result = result[idx]
            else:
                return None
        else:
            # Object key access, possibly mapped over array
            if isinstance(result, list):
                result = [item.get(part) if isinstance(item, dict) else None for item in result]
            elif isinstance(result, dict):
                result = result.get(part)
            else:
                return None

    return result


def csv_to_json(csv_content: str, delimiter: str = ",", has_header: bool = True) -> list[dict]:
    """Convert CSV string to list of dicts."""
    reader = csv.reader(io.StringIO(csv_content), delimiter=delimiter)
    rows = list(reader)

    if not rows:
        return []

    if has_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        headers = [f"col_{i}" for i in range(len(rows[0]))]
        data_rows = rows

    result = []
    for row in data_rows:
        obj = {}
        for i, value in enumerate(row):
            if i < len(headers):
                # Try to parse numbers
                try:
                    if "." in value:
                        obj[headers[i]] = float(value)
                    else:
                        obj[headers[i]] = int(value)
                except ValueError:
                    obj[headers[i]] = value
        result.append(obj)

    return result


def json_to_csv(data: list[dict], columns: list[str] | None = None) -> str:
    """Convert list of dicts to CSV string."""
    if not data:
        return ""

    if columns is None:
        # Collect all keys
        all_keys = set()
        for row in data:
            all_keys.update(row.keys())
        columns = sorted(all_keys)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(data)

    return output.getvalue()


def sql_query(query: str, tables: dict[str, list[dict]] | None = None) -> list[dict]:
    """Execute SQL on in-memory SQLite database.

    Args:
        query: SQL query to execute
        tables: Dict mapping table names to list of row dicts

    Returns:
        Query results as list of dicts
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create tables from provided data
    if tables:
        for table_name, rows in tables.items():
            if not rows:
                continue

            # Infer schema from first row
            columns = list(rows[0].keys())
            col_defs = ", ".join(f'"{col}" TEXT' for col in columns)
            conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

            # Insert data
            placeholders = ", ".join("?" for _ in columns)
            for row in rows:
                values = [row.get(col) for col in columns]
                conn.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', values)

    # Execute query
    try:
        cursor = conn.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        return results
    except sqlite3.Error as e:
        return [{"error": str(e)}]
    finally:
        conn.close()
