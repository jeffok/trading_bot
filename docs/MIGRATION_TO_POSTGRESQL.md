# MySQL/MariaDB到PostgreSQL迁移指南

## ✅ 已完成的工作

### 1. 基础架构
- ✅ 更新`.gitignore`，添加pyc、临时文件等
- ✅ 删除所有pyc文件和`__pycache__`目录
- ✅ 创建PostgreSQL适配器（`shared/db/postgres.py`）
- ✅ 更新`shared/db/__init__.py`，导出PostgreSQL
- ✅ 更新`shared/config/loader.py`，使用`POSTGRES_URL`

### 2. 配置
- ✅ 更新`.env.example`，使用PostgreSQL配置

## ⏳ 待完成的工作

### 1. 代码替换（需要批量替换）

#### 1.1 替换MariaDB初始化
需要将所有：
```python
db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
```

替换为：
```python
db = PostgreSQL(settings.postgres_url)
```

或者（向后兼容）：
```python
from shared.db import PostgreSQL as MariaDB
db = MariaDB(settings.postgres_url)
```

#### 1.2 需要修改的文件列表
- `services/api_service/main.py` (3处)
- `services/data_syncer/main.py` (1处)
- `services/strategy_engine/main.py` (1处)
- `tools/admin_cli/__main__.py` (3处)
- `tools/admin_cli/smoke.py` (1处)
- `tools/diagnose_no_orders.py` (1处)
- `shared/domain/*.py` (多个文件)

### 2. SQL语法转换

PostgreSQL和MySQL的SQL语法差异需要处理：

#### 2.1 反引号转双引号
- MySQL: `` `column_name` ``
- PostgreSQL: `"column_name"`

已在`shared/db/postgres.py`中自动处理。

#### 2.2 ON DUPLICATE KEY UPDATE
- MySQL: `ON DUPLICATE KEY UPDATE column=value`
- PostgreSQL: `ON CONFLICT (key) DO UPDATE SET column=value`

需要手动检查和修改所有使用`ON DUPLICATE KEY UPDATE`的SQL。

#### 2.3 其他差异
- `LIMIT`语法相同
- `OFFSET`语法相同
- 日期函数可能需要调整

### 3. 集成小工具到admin_cli

#### 3.1 diagnose_no_orders.py
需要将`tools/diagnose_no_orders.py`的功能集成到`tools/admin_cli/__main__.py`中：

```python
# 在main()函数中添加：
p_diagnose = sub.add_parser("diagnose", help="诊断为什么没有下单")
p_diagnose.add_argument("--symbol", type=str, help="指定交易对（可选）")

# 在命令处理中添加：
if args.cmd == "diagnose":
    from tools.admin_cli.diagnose import run_diagnose
    raise SystemExit(run_diagnose(settings, symbol=getattr(args, "symbol", None)))
```

#### 3.2 self_check.py
需要将`tools/self_check.py`的功能集成到`tools/admin_cli/__main__.py`中：

```python
# 在main()函数中添加：
p_check = sub.add_parser("check", help="语法检查（compileall）")

# 在命令处理中添加：
if args.cmd == "check":
    import compileall
    import os
    ok = compileall.compile_dir(os.path.abspath("."), quiet=1)
    raise SystemExit(0 if ok else 1)
```

### 4. 删除MySQL/MariaDB相关文件

需要删除或更新以下文件：
- `scripts/backup_db.sh` - 需要改为PostgreSQL备份脚本
- `scripts/restore_db.sh` - 需要改为PostgreSQL恢复脚本
- `scripts/wait_for_db.sh` - 需要更新端口检查（5432而不是3306）
- `scripts/drills/e2e_trade_cycle.sh` - 需要更新数据库命令
- `shared/db/maria.py` - 可以删除（已由postgres.py替代）
- `docker-compose.yml.example` - 需要改为PostgreSQL服务

### 5. 更新依赖

#### 5.1 requirements.txt
- 移除：`pymysql==1.1.1`
- 添加：`psycopg2-binary==2.9.9`（或更新版本）

#### 5.2 Dockerfile
确保所有服务的Dockerfile都包含`psycopg2-binary`。

### 6. 数据迁移

需要创建数据迁移脚本，将现有MySQL/MariaDB数据迁移到PostgreSQL：
1. 导出MySQL数据
2. 转换数据格式（JSON字段等）
3. 导入PostgreSQL

## 🔧 快速修复脚本

### 批量替换MariaDB初始化

```bash
# 查找所有需要替换的文件
grep -r "MariaDB(settings\." --include="*.py" .

# 手动替换每个文件中的：
# 旧：MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
# 新：PostgreSQL(settings.postgres_url)
```

### 批量替换SQL反引号

已在`shared/db/postgres.py`中自动处理，但建议检查所有SQL文件。

## 📝 测试清单

迁移完成后，需要测试：
- [ ] 数据库连接
- [ ] 数据读写
- [ ] 事务处理
- [ ] 迁移脚本执行
- [ ] 所有服务启动
- [ ] admin_cli所有命令
- [ ] API接口

## ⚠️ 注意事项

1. **向后兼容**：`MariaDB`类名保留为`PostgreSQL`的别名，确保代码可以逐步迁移
2. **SQL语法**：某些复杂的SQL可能需要手动调整
3. **数据迁移**：生产环境迁移前必须备份数据
4. **测试**：在测试环境充分测试后再部署到生产环境

## 🚀 下一步

1. 完成代码替换（使用上面的指南）
2. 更新所有SQL语句
3. 集成小工具到admin_cli
4. 删除MySQL/MariaDB相关文件
5. 更新依赖和Docker配置
6. 创建数据迁移脚本
7. 全面测试
