from .postgres import PostgreSQL
from .migrations import migrate

# 向后兼容：MariaDB别名指向PostgreSQL
MariaDB = PostgreSQL
