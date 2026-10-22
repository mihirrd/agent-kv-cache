import pytest

from kvtier.predictor import PausePredictor, percentile, tool_category


def test_percentile_interpolates():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0.5) == 3.0
    assert percentile(xs, 0.1) == pytest.approx(1.4)
    assert percentile(xs, 0.9) == pytest.approx(4.6)
    assert percentile([7.0], 0.9) == 7.0


@pytest.mark.parametrize(
    ("program", "category"),
    [("pytest", "test"), ("runtests.py", "test"), ("cargo", "build"), ("go", "build"), ("pip", "install"),
     ("grep", "inspect"), ("sed", "inspect"), ("python", "run"), ("python3", "run"), ("frobnicate", "other"),
     ("", "other")],
)
def test_tool_category(program, category):
    assert tool_category(program) == category


def test_no_prediction_until_min_samples():
    p = PausePredictor(min_samples=5)
    for _ in range(4):
        p.observe("s1", "pytest", 10.0)
    assert p.predict("s1", "pytest") is None
    p.observe("s1", "pytest", 10.0)
    pred = p.predict("s1", "pytest")
    assert pred.key == ("session", "s1", "pytest") and pred.n == 5 and pred.p50 == 10.0


def test_back_off_to_program_then_tool_then_global():
    p = PausePredictor(min_samples=3)
    for s in ("a", "b", "c"):
        p.observe(s, "pytest", 8.0)  # one sample per session: only the program key reaches 3
    assert p.predict("new", "pytest").key == ("program", "pytest")
    p.observe("a", "tox", 20.0)
    assert p.predict("new", "tox").key == ("tool", "test")  # tox unseen 3x, but "test" has 4 samples
    assert p.predict("new", "frobnicate").key == ("global",)


def test_ring_buffer_keeps_last_window():
    p = PausePredictor(window=50, min_samples=5)
    for _ in range(50):
        p.observe("s", "go", 100.0)
    for _ in range(50):
        p.observe("s", "go", 1.0)
    pred = p.predict("s", "go")
    assert pred.n == 50 and pred.p90 == 1.0


def test_confidence_rule():
    p = PausePredictor(min_samples=5, max_spread=20.0)
    for x in (1.0, 1.2, 1.5, 2.0, 3.0):
        p.observe("tight", "grep", x)
    assert p.predict("tight", "grep").confident
    for x in (0.01, 0.02, 5.0, 50.0, 60.0):
        p.observe("wide", "make", x)
    pred = p.predict("wide", "make")
    assert pred.p90 / pred.p10 > 20 and not pred.confident
    for x in (0.0, 0.0, 0.0, 0.0, 0.0):
        p.observe("zero", "echo", x)
    assert p.predict("zero", "echo").confident  # all-zero pauses: no spread, no division by zero


def test_remaining_time_given_elapsed():
    p = PausePredictor(min_samples=5)
    for x in (0.05, 0.05, 0.05, 0.05, 10.0, 30.0):
        p.observe("s", "git", x)
    assert p.remaining("s", "git", elapsed=0.0) == pytest.approx(0.05)  # median of all six
    assert p.remaining("s", "git", elapsed=1.0) == pytest.approx(19.0)  # only 10 and 30 outlast 1 s
    assert p.remaining("s", "git", elapsed=40.0) == float("inf")  # longer than anything seen
    assert p.remaining("new", "unseen", elapsed=1.0) == pytest.approx(19.0)  # global back-off
    assert PausePredictor().remaining("s", "git", 1.0) is None  # no history at all
