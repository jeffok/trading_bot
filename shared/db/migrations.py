
"""Simple migrations runner."""

from __future__ import annotations
import re
from pathlib import Path
from typing import List
from .postgres import PostgreSQL

MIGRATION_RE = re.compile(r"^(\d{4})_.*\.sql$")

def migrate(db: PostgreSQL, migrations_dir: Path) -> List[str]:
    migrations_dir = migrations_dir.resolve()
    with db.tx() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version VARCHAR(32) PRIMARY KEY,
              applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    applied = {r["version"] for r in db.fetch_all("SELECT version FROM schema_migrations")}
    ran: List[str] = []

    for p in sorted(migrations_dir.glob("*.sql")):
        m = MIGRATION_RE.match(p.name)
        if not m:
            continue
        version = m.group(1)
        if version in applied:
            continue

        raw_sql = p.read_text(encoding="utf-8")

        # IMPORTANT:
        # Migration files may contain:
        # 1. Comments that include semicolons
        # 2. PostgreSQL dollar-quoted strings ($$...$$) which may contain semicolons
        # A naive `split(';')` breaks these into executable fragments and causes SQL syntax errors.
        #
        # We implement a smarter parser:
        # - Remove /* ... */ block comments
        # - Remove full-line `-- ...` comments
        # - Remove inline `-- ...` comments (best-effort)
        # - Preserve dollar-quoted strings ($$...$$) when splitting by semicolon

        # Remove C-style block comments.
        sql = re.sub(r"/\*.*?\*/", "", raw_sql, flags=re.DOTALL)

        cleaned_lines = []
        for line in sql.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("--"):
                continue
            # Remove inline `-- comment` if present.
            # PostgreSQL requires a space after `--` for comment, but we treat any `--` as comment starter.
            if "--" in line:
                line = line.split("--", 1)[0]
            if line.strip():
                cleaned_lines.append(line)

        sql = "\n".join(cleaned_lines)
        
        # Split SQL statements by semicolon, but preserve dollar-quoted strings
        # Dollar-quoted strings can be $$...$$, $tag$...$tag$, etc.
        # Use character-by-character parsing to correctly handle nested cases
        
        statements = []
        current_statement = []
        i = 0
        in_dollar_quote = False
        dollar_tag = None  # None for $$, or the tag name for $tag$
        
        while i < len(sql):
            char = sql[i]
            
            if not in_dollar_quote:
                # Check for start of dollar quote
                if char == '$':
                    # Look ahead to see if this is a dollar quote start
                    # Match pattern: $tag$ or $$
                    j = i + 1
                    tag_chars = []
                    while j < len(sql) and sql[j] != '$':
                        tag_chars.append(sql[j])
                        j += 1
                    
                    if j < len(sql):  # Found closing $
                        tag = ''.join(tag_chars)
                        # This is a dollar quote start: $tag$ or $$
                        in_dollar_quote = True
                        dollar_tag = tag
                        current_statement.append(sql[i:j+1])  # Include $tag$ or $$
                        i = j + 1
                        continue
                
                # Not in dollar quote, check for statement separator
                if char == ';':
                    stmt = ''.join(current_statement).strip()
                    if stmt:
                        statements.append(stmt)
                    current_statement = []
                    i += 1
                    continue
            else:
                # We're inside a dollar-quoted string
                # Look for the closing tag
                if char == '$':
                    # Check if this matches the closing tag
                    if dollar_tag:
                        # Tagged dollar quote: $tag$...$tag$
                        closing_tag = f'${dollar_tag}$'
                        if i + len(closing_tag) <= len(sql) and sql[i:i+len(closing_tag)] == closing_tag:
                            # Found closing tag
                            current_statement.append(closing_tag)
                            i += len(closing_tag)
                            in_dollar_quote = False
                            dollar_tag = None
                            continue
                    else:
                        # Simple dollar quote: $$...$$
                        if i + 1 < len(sql) and sql[i+1] == '$':
                            # Found closing $$
                            current_statement.append('$$')
                            i += 2
                            in_dollar_quote = False
                            dollar_tag = None
                            continue
            
            # Regular character, add to current statement
            current_statement.append(char)
            i += 1
        
        # Add any remaining statement
        if current_statement:
            stmt = ''.join(current_statement).strip()
            if stmt:
                statements.append(stmt)
        with db.tx() as cur:
            for st in statements:
                cur.execute(st)
            cur.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (version,))
        ran.append(p.name)

    return ran
