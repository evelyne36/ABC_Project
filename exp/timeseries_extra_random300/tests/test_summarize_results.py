from __future__ import annotations

import pandas as pd

from timeseries_extra_random300.scripts.summarize_results import summarize


def test_summarize_falls_back_when_tabulate_is_missing(tmp_path, monkeypatch) -> None:
    summary_csv = tmp_path / "summary.csv"
    summary_csv.write_text(
        "\n".join(
            [
                "dataset,model,outer_fold,test_rmse,test_mae,total_train_time_sec,total_predict_time_sec",
                "ETTh1,xgb_cpu,1,0.7,0.5,1.0,0.1",
                "ETTh1,lgbm_cpu,1,0.6,0.4,1.1,0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def missing_tabulate(self, *args, **kwargs):
        raise ImportError("Missing optional dependency 'tabulate'.")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", missing_tabulate)

    report = summarize(tmp_path)

    assert "lgbm_cpu" in report
    assert (tmp_path / "aggregate.csv").exists()
    assert (tmp_path / "smoke_report.md").exists()
