"""After a deploy, app.py re-imports lookroom; loaded models must be reused, never loaded a second time (OOM)."""
import importlib
import sys


def test_models_survive_a_lookroom_reimport():
    import lookroom.llm as llm
    import lookroom.vision as vision

    sentinel = object()
    saved = {n: m for n, m in sys.modules.items() if n == "lookroom" or n.startswith("lookroom.")}
    old_model = llm._S["model"]
    llm._S["model"] = sentinel
    vision._taggers["probe-test"] = sentinel
    try:
        for n in saved:
            sys.modules.pop(n)
        new_llm = importlib.import_module("lookroom.llm")
        new_vision = importlib.import_module("lookroom.vision")
        assert new_llm is not llm and new_llm._S["model"] is sentinel
        assert new_vision._taggers.get("probe-test") is sentinel
    finally:
        llm._S["model"] = old_model
        vision._taggers.pop("probe-test", None)
        sys.modules.update(saved)
