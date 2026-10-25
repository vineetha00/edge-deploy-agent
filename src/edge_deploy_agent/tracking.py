"""Experiment tracking: one MLflow parent run per agent session, one nested run per trial."""
from __future__ import annotations

from typing import Any, Protocol


class Tracker(Protocol):
    def start(self, goal: dict) -> None: ...
    def log_trial(self, index: int, trial: dict, goal: dict) -> None: ...
    def finish(self, state: dict) -> None: ...


class NullTracker:
    def start(self, goal: dict) -> None:
        pass

    def log_trial(self, index: int, trial: dict, goal: dict) -> None:
        pass

    def finish(self, state: dict) -> None:
        pass


def trial_record(trial: dict, goal: dict) -> tuple[dict, dict, dict]:
    """Split a trial into (params, metrics, tags) for logging."""
    from .agent import meets_budget  # local import: agent imports this module

    p = trial.get("proposal") or {}
    params = {k: p[k] for k in ("model", "threads", "iters", "opt") if k in p}
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}
    if "blocked" in trial:
        tags["outcome"] = "blocked"
        tags["blocked_reason"] = str(trial["blocked"])[:250]
    elif "error" in trial:
        tags["outcome"] = "error"
        tags["error"] = str(trial["error"])[:250]
    else:
        r = trial["result"]
        lat = r.get("latency_ms", {})
        for k in ("p50", "p95", "mean"):
            if k in lat:
                metrics[f"latency_{k}_ms"] = float(lat[k])
        for k in ("peak_rss_mb", "temp_c_before", "temp_c_after", "cooldown_s"):
            if isinstance(r.get(k), int | float):
                metrics[k] = float(r[k])
        wer = (goal.get("wer") or {}).get(p.get("model", ""))
        if wer is not None:
            metrics["wer"] = float(wer)
        tags["outcome"] = "ran"
        tags["run_valid"] = str(r.get("run_valid"))
        tags["meets_budget"] = str(meets_budget(r, goal))
    return params, metrics, tags


class MlflowTracker:
    def __init__(self, tracking_uri: str, experiment: str = "edge-deploy-agent",
                 run_name: str | None = None, artifact_location: str | None = None):
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        if artifact_location and mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=artifact_location)
        mlflow.set_experiment(experiment)
        self._run_name = run_name
        self.parent_run_id: str | None = None

    def start(self, goal: dict) -> None:
        mlflow = self._mlflow
        run = mlflow.start_run(run_name=self._run_name)
        self.parent_run_id = run.info.run_id
        mlflow.log_params({
            "candidates": ",".join(goal.get("candidates", [])),
            "budget_p95_ms": goal.get("latency_p95_ms"),
            "budget_ram_mb": goal.get("ram_mb"),
            "max_wer": goal.get("max_wer"),
            "max_iters": goal.get("max_iters"),
        })

    def log_trial(self, index: int, trial: dict, goal: dict) -> None:
        mlflow = self._mlflow
        params, metrics, tags = trial_record(trial, goal)
        name = f"trial-{index:02d}-{params.get('model', 'na')}"
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.set_tags(tags | {"trial_index": str(index)})

    def finish(self, state: dict) -> None:
        mlflow = self._mlflow
        best: dict[str, Any] | None = state.get("best")
        if best:
            r = best["result"]
            mlflow.set_tag("best_model", best["proposal"].get("model"))
            mlflow.log_metrics({"best_latency_p95_ms": float(r["latency_ms"]["p95"]),
                                "best_peak_rss_mb": float(r["peak_rss_mb"])})
        trials = state.get("trials", [])
        mlflow.log_metrics({"n_trials": len(trials),
                            "n_blocked": sum(1 for t in trials if "blocked" in t)})
        mlflow.log_dict({"trials": trials, "best": best}, "trials.json")
        mlflow.log_text(state.get("report") or "", "report.txt")
        mlflow.end_run()
