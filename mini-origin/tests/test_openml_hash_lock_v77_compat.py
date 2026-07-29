import json
import sys
from types import SimpleNamespace

from mini_origin import openml_hash_lock_v77_compat as compat


def test_compatibility_shim_supplies_only_preregistered_suite_id(monkeypatch):
    suite = SimpleNamespace(alias="OpenML-CC18")
    fake_openml = SimpleNamespace(
        study=SimpleNamespace(get_suite=lambda suite_id: suite)
    )
    monkeypatch.setitem(sys.modules, "openml", fake_openml)

    expected = json.loads(
        compat.lock.PREREGISTRATION.read_text(encoding="utf-8-sig")
    )["benchmark_suite_id"]
    installed = compat.install_suite_id_compatibility()
    returned = fake_openml.study.get_suite(expected)

    assert installed == expected == 99
    assert returned is suite
    assert returned.suite_id == 99


def test_frozen_selection_module_is_not_modified_by_compatibility_shim():
    source = compat.lock.Path(compat.lock.__file__).read_text(encoding="utf-8")
    assert '"suite_id": int(suite.suite_id)' in source
    assert "selection_seed" in source
