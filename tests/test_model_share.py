"""Which copy of the model a device trusts, and where fetched models live."""

from __future__ import annotations

import pytest

from torrent_llm.net import paths
from torrent_llm.pool.model_share import COMPLETE, _choose, local_model_dir, shared_dir

A, B = "a" * 64, "b" * 64


def test_a_pinned_manifest_is_the_only_one_accepted():
    assert _choose({"p1": B, "p2": A}, public=True, pinned=A) == A
    with pytest.raises(LookupError, match="exact files"):
        _choose({"p1": B}, public=False, pinned=A)


def test_an_invite_swarm_takes_the_first_offer():
    assert _choose({"p1": A, "p2": B}, public=False, pinned=None) == A


def test_the_public_swarm_needs_devices_to_agree():
    with pytest.raises(LookupError, match="2 must agree"):
        _choose({"p1": A, "p2": B}, public=True, pinned=None)
    assert _choose({"p1": A, "p2": B, "p3": B}, public=True, pinned=None) == B


def test_no_offer_is_a_lookup_error():
    with pytest.raises(LookupError):
        _choose({}, public=False, pinned=None)


def test_fetched_models_are_found_only_when_complete(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_OVERRIDE, str(tmp_path))
    where = shared_dir("Org/Model-1")
    assert where.parent == tmp_path / "models"
    assert where.name == "Org--Model-1"
    where.mkdir(parents=True)
    assert local_model_dir("Org/Model-1") is None  # not finished, and not in the hub cache
    (where / COMPLETE).write_text("x")
    assert local_model_dir("Org/Model-1") == where


def test_odd_model_ids_stay_inside_the_models_folder(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_OVERRIDE, str(tmp_path))
    for model_id in ("..", "../../etc", "a/../../b", "C:\\x", "", "x" * 500):
        where = shared_dir(model_id)
        assert where.parent == tmp_path / "models"
        assert where.name not in ("", ".", "..")
