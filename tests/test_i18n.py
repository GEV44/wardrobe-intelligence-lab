import ast
from pathlib import Path

from lookroom import i18n

SRC = Path(i18n.__file__)


def test_no_duplicate_keys_in_string_tables():
    # a duplicated key in a dict literal silently overrides the first one
    for node in ast.walk(ast.parse(SRC.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            assert len(keys) == len(set(keys)), {k for k in keys if keys.count(k) > 1}


def test_every_ui_string_has_all_languages():
    for lang in ("en", "hy", "ru"):
        assert i18n.t("another", lang) and i18n.t("biggest_gap", lang)
