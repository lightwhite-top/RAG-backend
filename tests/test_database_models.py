"""数据库模型建表选项测试。"""

from baozhi_rag.infra.database.models import Base


def test_all_mysql_tables_lock_charset_and_collation() -> None:
    """所有 ORM 表都应显式锁定字符集与排序规则。

    返回:
        无返回值。

    异常:
        AssertionError: 当任一表未显式声明统一的 MySQL 字符集或排序规则时抛出。
    """
    for table in Base.metadata.sorted_tables:
        mysql_options = table.dialect_options["mysql"]

        # 显式锁定排序规则，避免外键列继承数据库默认值后与历史表结构冲突。
        assert mysql_options["charset"] == "utf8mb4"
        assert mysql_options["collate"] == "utf8mb4_unicode_ci"
