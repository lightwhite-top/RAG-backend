"""校验提交信息是否符合 Conventional Commits。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

COMMIT_PATTERN = re.compile(
    r"^(feat|fix|refactor|docs|test|chore|build|ci|perf|revert)"
    r"(\([a-z0-9_-]+\))?: .{1,72}$"
)


def main() -> int:
    """读取提交信息并执行格式校验。"""
    if len(sys.argv) != 2:
        print("用法: python scripts/check_commit_msg.py <commit_msg_file>")
        return 1

    commit_file = Path(sys.argv[1])
    lines = commit_file.read_text(encoding="utf-8").splitlines()
    if not lines:
        print("提交信息不能为空。")
        return 1

    first_line = lines[0].strip()

    if COMMIT_PATTERN.match(first_line):
        return 0

    print("提交信息不符合规范。示例: feat(api): 初始化健康检查接口")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
