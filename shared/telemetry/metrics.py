from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


class Metrics:
    """Prometheus metrics (shared across services).

    Note: prometheus_client uses a global default registry. Each service process must expose /metrics
    (FastAPI route or start_http_server) to make these visible.
    """

    def __init__(self, service: str):
        self.service = service

        # Leader election (HA)
        self.leader_is_leader = Gauge(
            "leader_is_leader",
            "Whether this instance is the leader (1 leader, 0 follower)",
            ("service", "instance_id"),
        )
        self.leader_changes_total = Counter(
            "leader_changes_total",
            "Leadership role changes",
            ("service", "instance_id", "role"),
        )


        # Trading / orders
        self.orders_total = Counter(
            "orders_total",
            "Total orders (by final status label)",
            ("service", "exchange", "symbol", "status"),
        )

        # Exchange I/O
        self.exchange_requests_total = Counter(
            "exchange_requests_total",
            "Total exchange HTTP/API requests",
            ("service", "exchange", "endpoint", "status"),
        )
        self.exchange_latency_seconds = Histogram(
            "exchange_latency_seconds",
            "Exchange request latency (seconds)",
            ("service", "exchange", "endpoint"),
        )

        # Data sync
        self.data_sync_lag_ms = Gauge(
            "data_sync_lag_ms",
            "Data sync lag (ms) between now and last cached kline open_time_ms",
            ("service", "symbol", "interval_minutes"),
        )
        self.data_sync_cycles_total = Counter(
            "data_sync_cycles_total",
            "Data sync cycles total",
            ("service",),
        )
        self.data_sync_errors_total = Counter(
            "data_sync_errors_total",
            "Data sync errors total",
            ("service",),
        )
        self.data_sync_gaps_total = Counter(
            "data_sync_gaps_total",
            "Detected kline gaps total",
            ("service", "symbol", "interval_minutes"),
        )


        self.data_sync_gap_fill_runs_total = Counter(
            "data_sync_gap_fill_runs_total",
            "Gap fill runs total",
            ("service", "symbol", "interval_minutes"),
        )
        self.data_sync_gap_fill_bars_total = Counter(
            "data_sync_gap_fill_bars_total",
            "Gap fill bars inserted total",
            ("service", "symbol", "interval_minutes"),
        )

        # Precompute / feature cache
        self.precompute_tasks_enqueued_total = Counter(
            "precompute_tasks_enqueued_total",
            "Precompute tasks enqueued total",
            ("service", "symbol", "interval_minutes"),
        )
        self.precompute_tasks_processed_total = Counter(
            "precompute_tasks_processed_total",
            "Precompute tasks processed total",
            ("service", "symbol", "interval_minutes"),
        )
        self.precompute_errors_total = Counter(
            "precompute_errors_total",
            "Precompute errors total",
            ("service", "symbol", "interval_minutes"),
        )
        self.feature_compute_seconds = Histogram(
            "feature_compute_seconds",
            "Feature computation duration (seconds)",
            ("service", "symbol"),
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        )

        # Strategy ticks
        self.last_tick_success = Gauge(
            "last_tick_success",
            "Last tick success flag (1=ok, 0=fail)",
            ("service", "symbol"),
        )
        self.tick_duration_seconds = Histogram(
            "tick_duration_seconds",
            "Tick duration (seconds)",
            ("service", "symbol"),
        )
        self.tick_errors_total = Counter(
            "tick_errors_total",
            "Tick errors total",
            ("service", "symbol"),
        )

        # Reconciliation
        self.reconcile_runs_total = Counter(
            "reconcile_runs_total",
            "Reconciliation runs total",
            ("service",),
        )
        self.reconcile_orders_total = Counter(
            "reconcile_orders_total",
            "Reconciliation checked orders total",
            ("service", "symbol"),
        )
        self.reconcile_fixed_total = Counter(
            "reconcile_fixed_total",
            "Reconciliation fixed/closed orders total",
            ("service", "symbol", "final_status"),
        )

        # Archival
        self.archive_runs_total = Counter(
            "archive_runs_total",
            "Archive runs total",
            ("service",),
        )
        self.archive_rows_total = Counter(
            "archive_rows_total",
            "Archived rows total",
            ("service", "table_name"),
        )
        self.archive_errors_total = Counter(
            "archive_errors_total",
            "Archive errors total",
            ("service",),
        )


        # Trades (lifecycle)
        self.trades_open_total = Counter(
            "trades_open_total",
            "Trades opened total",
            ("service", "symbol"),
        )
        self.trades_close_total = Counter(
            "trades_close_total",
            "Trades closed total",
            ("service", "symbol", "close_reason_code"),
        )
        self.trade_last_pnl_usdt = Gauge(
            "trade_last_pnl_usdt",
            "Last trade pnl (usdt)",
            ("service", "symbol"),
        )
        self.trade_last_duration_seconds = Gauge(
            "trade_last_duration_seconds",
            "Last trade holding duration (seconds)",
            ("service", "symbol"),
        )

        # AI
        self.ai_predictions_total = Counter(
            "ai_predictions_total",
            "AI predictions total",
            ("service", "symbol"),
        )
        self.ai_training_total = Counter(
            "ai_training_total",
            "AI training updates total",
            ("service", "symbol"),
        )
        self.ai_model_seen = Gauge(
            "ai_model_seen",
            "AI model seen samples",
            ("service",),
        )
