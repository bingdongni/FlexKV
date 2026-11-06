"""Native CPU radix/SWA check for prefetch terminal metrics."""

from test_planner_native import env, finish_swa, start, swa_env
from test_prefetch_metrics import metrics, sample


def test_native_swa_failed_tail_counts_only_the_last_checkpoint(swa_env, metrics):
    collector, registry = metrics
    swa_env.c.metrics = collector
    handle = start(swa_env, swa_aware=True)
    swa_env.c.tick()
    finish_swa(swa_env, swa_env.sent[0])
    finish_swa(swa_env, swa_env.sent[1])
    swa_env.c.tick()
    swa_env.c.tick()
    swa_env.c.stop(handle, "demand")
    for graph in list(swa_env.sent)[2:]:
        finish_swa(swa_env, graph, snapshot=False)
    swa_env.c.tick()
    snapshot = swa_env.c.snapshot(handle)
    assert snapshot.terminal and snapshot.loaded_tokens == 16
    assert snapshot.stop_reason == "partial_read" and snapshot.outcome == "partial"
    assert sample(registry, "flexkv_prefetch_loaded_tokens_total", policy="wait_complete") == 16
    assert sample(registry, "flexkv_prefetch_sessions_total",
                  policy="wait_complete", reason="partial_read", outcome="partial") == 1
