from app.observability.stats import Series, StatsRegistry, percentile

TEN = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]   # 排序后 1 1 2 3 3 4 5 5 6 9


def test_1_empty():
    assert percentile([], 95) == 0.0


def test_2_input_is_not_mutated():
    values = [3, 1, 2]
    percentile(values, 50)
    assert values == [3, 1, 2]


def test_3_worked_examples_from_docstring():
    v = list(range(1, 11))
    assert (percentile(v, 50), percentile(v, 95), percentile(v, 0), percentile(v, 100)) == (5, 10, 1, 10)


def test_3_unsorted_input():
    assert percentile(TEN, 50) == 3 and percentile(TEN, 90) == 6 and percentile(TEN, 99) == 9


def test_3_single_value():
    assert percentile([7.5], 50) == 7.5 and percentile([7.5], 99) == 7.5


def test_series_summary():
    s = Series()
    for ms, ok in [(100, True), (200, True), (5000, False)]:
        s.record(ms, ok)
    out = s.summary()
    assert (out["total"], out["failures"], out["success_rate"]) == (3, 1, 0.6667)
    assert out["p50_ms"] == 200 and out["max_ms"] == 5000


def test_registry_groups_by_kind_and_name():
    r = StatsRegistry()
    r.record("agent", "billing", 120)
    r.record("agent", "billing", 80, ok=False)
    r.record("tool", "get_order_status", 3)
    snap = r.snapshot()
    assert snap["agent"]["billing"]["total"] == 2 and snap["agent"]["billing"]["failures"] == 1
    assert snap["tool"]["get_order_status"]["p50_ms"] == 3
