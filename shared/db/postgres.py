"""PostgreSQL helper (minimal, safe)

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
    from psycopg2 import extensions
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    ThreadedConnectionPool = None
    extensions = None


class PostgreSQL:
    """PostgreSQL数据库适配器"""
    
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
            database = parsed.path.lstrip('/')
            
            # 如果数据库不存在，尝试创建（连接到postgres数据库）
            if database:
                try:
                    # 先尝试连接目标数据库
                    test_conn = psycopg2.connect(
                        host=parsed.hostname,
                        port=parsed.port or 5432,
                        user=parsed.username,
                        password=parsed.password,
                        database=database,
                        connect_timeout=2,
                    )
                    test_conn.close()
                except psycopg2.OperationalError as e:
                    if "does not exist" in str(e):
                        # 数据库不存在，尝试创建
                        try:
                            admin_conn = psycopg2.connect(
                                host=parsed.hostname,
                                port=parsed.port or 5432,
                                user=parsed.username,
                                password=parsed.password,
                                database="postgres",  # 连接到默认数据库
                                connect_timeout=2,
                            )
                            admin_conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                            cur = admin_conn.cursor()
                            # 使用双引号转义数据库名
                            cur.execute(f'CREATE DATABASE "{database}"')
                            cur.close()
                            admin_conn.close()
                        except Exception as create_err:
                            # 创建失败，抛出原始错误
                            raise e
            
            self._pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=database,
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
        with self.tx() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row:
                return dict(row)
            return None
    
    def fetch_all(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        """执行查询并返回所有结果"""
        with self.tx() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    
    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> int:
        """执行SQL并返回影响的行数"""
        with self.tx() as cur:
            cur.execute(sql, params)
            return cur.rowcount
    
    def close(self):
        """关闭连接池"""
        if self._pool:
            self._pool.closeall()
            self._pool = None
