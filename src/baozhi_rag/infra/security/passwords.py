"""密码哈希适配。"""

from __future__ import annotations

from functools import lru_cache

from pwdlib import PasswordHash


class PasswordHasherAdapter:
    """封装密码哈希与校验逻辑。"""

    def __init__(self, password_hash: PasswordHash) -> None:
        """初始化密码哈希器。"""
        self._password_hash = password_hash

    def hash_password(self, password: str) -> str:
        """生成密码哈希。"""
        return self._password_hash.hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        """校验明文密码是否匹配。"""
        return self._password_hash.verify(password, password_hash)

    @classmethod
    def from_default(cls) -> PasswordHasherAdapter:
        """返回缓存后的默认密码哈希器。"""
        return _build_default_password_hasher()


@lru_cache(maxsize=1)
def _build_default_password_hasher() -> PasswordHasherAdapter:
    """构造默认的密码哈希器。"""
    return PasswordHasherAdapter(PasswordHash.recommended())
