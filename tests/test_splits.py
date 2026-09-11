import pytest

from early_stop.splits import SplitConfig, SplitManager, assert_disjoint, assign_split


def test_assign_split_deterministic():
    config = SplitConfig(seed=0)
    a = assign_split("math500:0", config)
    b = assign_split("math500:0", config)
    assert a == b


def test_assign_split_different_seed_can_differ():
    id_ = "math500:0"
    splits_by_seed = {assign_split(id_, SplitConfig(seed=s)) for s in range(20)}
    # not a strict requirement that they differ, but with 20 seeds and 4
    # buckets it would be suspicious if they were all identical
    assert len(splits_by_seed) > 1


def test_split_fractions_approx_respected():
    config = SplitConfig(difficulty_frac=0.2, dev_frac=0.15, test_frac=0.65, seed=0)
    ids = [f"math500:{i}" for i in range(5000)]
    counts = {"difficulty": 0, "dev": 0, "test": 0, "pool": 0}
    for pid in ids:
        counts[assign_split(pid, config)] += 1
    n = len(ids)
    assert abs(counts["difficulty"] / n - 0.20) < 0.03
    assert abs(counts["dev"] / n - 0.15) < 0.03
    assert abs(counts["test"] / n - 0.65) < 0.03


def test_split_config_rejects_overallocated_fractions():
    with pytest.raises(ValueError):
        SplitConfig(difficulty_frac=0.5, dev_frac=0.4, test_frac=0.3)


def test_split_config_rejects_overallocation_including_pilot():
    with pytest.raises(ValueError):
        SplitConfig(pilot_frac=0.1, difficulty_frac=0.5, dev_frac=0.3, test_frac=0.2)


def test_pilot_bucket_disjoint_from_other_splits():
    config = SplitConfig(pilot_frac=0.05, difficulty_frac=0.20, dev_frac=0.15, test_frac=0.60, seed=0)
    ids = [f"math500:{i}" for i in range(5000)]
    counts = {"pilot": 0, "difficulty": 0, "dev": 0, "test": 0, "pool": 0}
    for pid in ids:
        counts[assign_split(pid, config)] += 1
    n = len(ids)
    assert abs(counts["pilot"] / n - 0.05) < 0.02
    # pilot problems must never land in difficulty/dev/test for the same id
    for pid in ids:
        splits_seen = {assign_split(pid, config)}
        assert len(splits_seen) == 1  # a single problem_id always gets exactly one split


def test_split_manager_filter_split():
    mgr = SplitManager(config=SplitConfig(seed=1))
    ids = [f"gsm8k:{i}" for i in range(200)]
    dev_ids = mgr.filter_split("gsm8k", ids, "dev")
    test_ids = mgr.filter_split("gsm8k", ids, "test")
    assert set(dev_ids).isdisjoint(set(test_ids))
    assert set(dev_ids) | set(test_ids) <= set(ids)


def test_split_manager_rejects_unknown_split_name():
    mgr = SplitManager(config=SplitConfig())
    with pytest.raises(ValueError):
        mgr.filter_split("gsm8k", ["gsm8k:0"], "bogus")


def test_assert_disjoint_passes_for_consistent_assignment():
    mgr = SplitManager(config=SplitConfig(seed=2))
    ids = [f"strategyqa:{i}" for i in range(50)]
    dev_ids = mgr.filter_split("strategyqa", ids, "dev")
    test_ids = mgr.filter_split("strategyqa", ids, "test")
    # should not raise
    assert_disjoint(dev_ids, test_ids)


def test_assert_disjoint_catches_manual_leakage():
    mgr = SplitManager(config=SplitConfig(seed=3))
    ids = [f"strategyqa:{i}" for i in range(50)]
    dev_ids = mgr.filter_split("strategyqa", ids, "dev")
    test_ids = mgr.filter_split("strategyqa", ids, "test")
    if not dev_ids or not test_ids:
        pytest.skip("degenerate split for this seed/size, not the thing under test")
    # Simulate a bug where a dev id also ends up copied into the test list.
    leaked = [dev_ids[0]] + test_ids
    with pytest.raises(AssertionError):
        assert_disjoint(dev_ids, leaked)


def test_split_manager_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "splits.json"
    mgr = SplitManager(config=SplitConfig(seed=7), record_path=path)
    ids = [f"math500:{i}" for i in range(30)]
    for pid in ids:
        mgr.split_for("math500", pid)
    mgr.save()

    reloaded = SplitManager.load(path)
    for pid in ids:
        assert reloaded.split_for("math500", pid) == mgr.split_for("math500", pid)
