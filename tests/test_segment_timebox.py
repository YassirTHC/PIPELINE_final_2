import time as _time

def test_segment_timebox_run_with_timeout_limits_duration():
    from video_processor import run_with_timeout

    def _slow_task():
        _time.sleep(5)
        return "done"

    start = _time.perf_counter()
    res = run_with_timeout(_slow_task, 1.5)
    elapsed = (_time.perf_counter() - start) * 1000

    # Expect None due to timeout and elapsed under ~2 seconds
    assert res is None
    assert elapsed < 2000

