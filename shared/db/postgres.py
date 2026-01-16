"""PostgreSQL helper (minimal, safe) - 替换MariaDB。

使用psycopg2连接PostgreSQL，数据以JSONB格式存储。
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    ThreadedConnectionPool = None


class PostgreSQL:
    """PostgreSQL数据库适配器（兼容MariaDB接口）"""
    
    def __init__(self, postgres_url: str):
        """
        初始化PostgreSQL连接
        
        Args:
            postgres_url: PostgreSQL连接URL，格式：postgresql://user:password@host:port/dbname
        """
        if psycopg2 is None:
            raise ImportError("psycopg2 is required. Install it with: pip install psycopg2-binary")
        
        self.postgres_url = postgres_url
        self._pool: Optional[ThreadedConnectionPool] = None
    
    def _get_connection(self):
        """获取数据库连接"""
        if self._pool is None:
            # 解析URL
            parsed = urlparse(self.postgres_url)
            self._pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path.lstrip('/'),
            )
        return self._pool.getconn()
    
    def _return_connection(self, conn):
        """归还连接到池"""
        if self._pool:
            self._pool.putconn(conn)
    
    @contextlib.contextmanager
    def tx(self):
        """事务上下文管理器"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
    
    def ping(self) -> bool:
        """检查数据库连接"""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    cur.fetchone()
                conn.commit()
            finally:
                self._return_connection(conn)
            return True
        except Exception:
            return False
    
    def fetch_one(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        """执行查询并返回单行结果"""
        # 将MySQL的%s占位符转换为PostgreSQL的%s（PostgreSQL也支持%s）
        # 将MySQL的反引号转换为PostgreSQL的双引号
        sql = sql.replace('`', '"')
        with self.tx() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row:
                return dict(row)
            return None
    
    def fetch_all(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        """执行查询并返回所有结果"""
        # 将MySQL的反引号转换为PostgreSQL的双引号
        sql = sql.replace('`', '"')
        with self.tx() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    
    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> int:
        """执行SQL并返回影响的行数"""
        # 将MySQL的反引号转换为PostgreSQL的双引号
        sql = sql.replace('`', '"')
        
        # 将MySQL的ON DUPLICATE KEY UPDATE转换为PostgreSQL的ON CONFLICT
        # PostgreSQL需要指定冲突列，这里尝试从INSERT语句中提取主键
        if 'ON DUPLICATE KEY UPDATE' in sql.upper():
            import re
            # 尝试提取INSERT INTO table_name (columns)中的表名和列
            insert_match = re.search(r'INSERT\s+INTO\s+"?(\w+)"?\s*\(([^)]+)\)', sql, re.IGNORECASE)
            if insert_match:
                table_name = insert_match.group(1)
                columns = [c.strip().strip('"') for c in insert_match.group(2).split(',')]
                # 假设第一个列是主键（大多数情况下）
                # 对于system_config，key是主键；对于service_status，service_name+instance_id是主键
                if table_name == 'system_config' and 'key' in columns:
                    conflict_col = 'key'
                elif table_name == 'service_status' and 'service_name' in columns and 'instance_id' in columns:
                    conflict_col = '(service_name, instance_id)'
                else:
                    # 默认使用第一个列
                    conflict_col = columns[0] if columns else 'id'
                
                # 替换ON DUPLICATE KEY UPDATE
                # 将VALUES(column)转换为EXCLUDED.column
                update_part = sql.split('ON DUPLICATE KEY UPDATE', 1)[1]
                # 替换VALUES(column)为EXCLUDED.column
                update_part = re.sub(r'VALUES\s*\(([^)]+)\)', r'EXCLUDED.\1', update_part, flags=re.IGNORECASE)
                sql = sql.split('ON DUPLICATE KEY UPDATE', 1)[0]
                sql = f"{sql} ON CONFLICT ({conflict_col}) DO UPDATE SET {update_part}"
            else:
                # 如果无法解析，使用简化的转换（可能不工作，但至少不会报语法错误）
                sql = sql.replace('ON DUPLICATE KEY UPDATE', 'ON CONFLICT DO UPDATE SET')
                sql = sql.replace('VALUES(', 'EXCLUDED.')
        
        with self.tx() as cur:
            cur.execute(sql, params)
            return cur.rowcount
    
    def close(self):
        """关闭连接池"""
        if self._pool:
            self._pool.closeall()
            self._pool = None


# 向后兼容：MariaDB别名
MariaDB = PostgreSQL
