#!/usr/bin/env python3
"""轻量自检脚本（不引入 pytest / 不依赖数据库驱动）。

用法：
  python tools/self_check.py

说明：
- 只做语法/编译级别检查，避免在没有安装 pymysql 等依赖时失败。
"""
from __future__ import annotations

import compileall
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def main() -> int:
    ok = compileall.compile_dir(ROOT, quiet=1)
    if not ok:
        print("❌ compileall 失败：请检查语法错误")
        return 1
    print("✅ compileall 通过")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
