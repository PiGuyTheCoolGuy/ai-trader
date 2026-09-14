"""Portable HTML report, no web server, CDN, or external assets required."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import atomic_bytes


def table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    keys = list(rows[0])
    head = "".join(f"<th>{escape(k)}</th>" for k in keys)
    body = ""
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key)
            text = (
                "—"
                if value is None
                else f"{value:,.3f}"
                if isinstance(value, float)
                else str(value)
            )
            cells.append(f"<td>{escape(text)}</td>")
        body += "<tr>" + "".join(cells) + "</tr>"
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def chart(result: dict) -> str:
    frame = pd.DataFrame({"Strategy": result["equity"], "Buy and hold": result["buy_hold_equity"]})
    indexes = np.unique(np.linspace(0, len(frame) - 1, min(700, len(frame))).astype(int))
    points = frame.iloc[indexes]
    low, high = float(points.min().min()), float(points.max().max())
    span = high - low or 1
    content = ""
    for j in range(5):
        y, value = 30 + j * 62.5, high - j * span / 4
        content += f'<line x1="85" y1="{y}" x2="970" y2="{y}" stroke="#293a50"/>'
        content += f'<text x="5" y="{y + 5}" fill="#a8b9cc">{value:,.0f}</text>'
    for name, color in [("Buy and hold", "#7c92bd"), ("Strategy", "#50e3b5")]:
        coords = " ".join(
            f"{85 + i * 885 / max(1, len(points) - 1):.1f},{30 + (high - v) / span * 250:.1f}"
            for i, v in enumerate(points[name])
        )
        content += f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>'
    content += f'<text x="85" y="315" fill="#a8b9cc">{points.index[0].date()}</text>'
    content += f'<text x="870" y="315" fill="#a8b9cc">{points.index[-1].date()}</text>'
    return f'<svg viewBox="0 0 1000 340" role="img" aria-label="Held-out equity against buy and hold">{content}</svg>'


def render_report(run_dir: Path, report: dict, result: dict) -> None:
    m = result["metrics"]
    cards = [
        ("Test return", f"{m['return_pct']:.2f}%"),
        ("Maximum drawdown", f"{m['max_drawdown_pct']:.2f}%"),
        ("Closed trades", str(m["closed_trades"])),
        ("Fees paid", f"{m['fees_paid']:,.2f} USDT"),
    ]
    card_html = "".join(
        f'<div class="card"><small>{label}</small><strong>{value}</strong></div>'
        for label, value in cards
    )
    splits = table(
        [
            {
                "Period": name,
                "Rows": info["rows"],
                "First candle": info["first"],
                "Last label end": info["last_label_end"],
            }
            for name, info in report["splits"].items()
        ]
    )
    trials = table(
        [
            {
                k: t[k]
                for k in [
                    "trial",
                    "trees",
                    "best_iteration",
                    "rmse_bps",
                    "mean_baseline_rmse_bps",
                    "elapsed_seconds",
                ]
            }
            for t in report["trials"]
        ]
    )
    candidates = table(
        [
            {
                "Threshold (bps)": c["threshold_bps"],
                "Net return %": c["metrics"]["return_pct"],
                "Drawdown %": c["metrics"]["max_drawdown_pct"],
                "Trades": c["metrics"]["closed_trades"],
                "Eligible": c["eligible"],
            }
            for c in report["calibration_candidates"]
        ]
    )
    symbols = table(
        [
            {
                "Symbol": s,
                **{
                    k: x[k]
                    for k in ["return_pct", "closed_trades", "max_drawdown_pct", "fees_paid"]
                },
            }
            for s, x in result["by_symbol"].items()
        ]
    )
    importance = table(
        [
            {"Feature": k, "Gain": v}
            for k, v in sorted(
                report["feature_importance_gain"].items(), key=lambda x: x[1], reverse=True
            )[:15]
        ]
    )
    monthly = result["equity"].resample("ME").last()
    prior = monthly.shift(1).fillna(m["initial_equity"])
    months = table(
        [
            {
                "Month": str(t.date()),
                "Return %": float((v / prior.loc[t] - 1) * 100),
                "Ending equity": float(v),
            }
            for t, v in monthly.items()
        ]
    )
    total_rows = sum(d["candles"] for d in report["data"].values())
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Trader · Training report</title>
<style>body{{margin:0;background:#0c1421;color:#edf4fc;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:40px 24px 80px}}h1{{font-size:42px;margin:10px 0}}
h2{{margin-top:38px;font-size:23px}}p,small{{color:#a8b9cc}}.tag{{color:#50e3b5;letter-spacing:2px;font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
.card,.notice{{padding:20px;border:1px solid #293a50;background:#132033;border-radius:12px}}
strong{{display:block;font-size:29px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:10px 14px;border-bottom:1px solid #293a50;text-align:left;white-space:nowrap}}
th{{color:#8aa4c2}}svg{{width:100%;background:#132033;border-radius:12px}}
code{{color:#50e3b5}}a{{color:#50e3b5}}</style><main>
<div class="tag">CPU TRAINING / HISTORICAL RESEARCH / PAPER ONLY</div>
<h1>What did the model learn?</h1><p>Run {escape(report["run_id"])} · {total_rows:,} real or supplied candles ·
{report["feature_count"]} features · {report["chosen_trees"]} fitted trees · {report["threads"]} CPU threads ·
{report["elapsed_seconds"] / 60:.1f} minutes</p>
<div class="notice"><b>Training completed.</b> {escape(report["selection_reason"])}
<br>Chosen threshold: {escape(str(report["threshold_bps"]))} bps (null means cash).
Predictions represent estimated gross future returns, not probabilities.</div>
<h2>Untouched test period</h2><div class="cards">{card_html}</div>
<p>Green: strategy · blue: equal-allocation buy and hold. Both include configured trading costs.
Capital is divided equally among symbols; the total starting balance is {m["initial_equity"]:,.0f} USDT.</p>
{chart(result)}{symbols}<h2>Monthly test results</h2>{months}
<h2>Where the data went</h2>{splits}<p>{report["purged_rows"]:,} labeled rows were purged at split boundaries.
Training fits the trees; tuning selects model complexity; calibration selects the trading threshold;
test measures the already frozen strategy. No random train/test shuffling.</p>
<h2>CPU model trials</h2>{trials}<p>Early stopping can finish well before the round limit.
Runtime alone is not a measure of model quality.</p>
<h2>Why it traded—or stayed in cash</h2>{candidates}
<p>Minimum predicted return including configured costs and extra edge: {report["cost_floor_bps"]:.2f} bps.
A candidate must have enough closed trades, positive return after costs, no drawdown halt, and a
positive return-minus-half-drawdown score. The model may correctly produce no eligible strategy.</p>
<h2>Features used by the trees</h2>{importance}
<h2>Simulation assumptions</h2><p>Signals use closed candles and fill at the next available candle open.
If stop and target are both touched inside a bar, the stop wins. Each symbol has a separate capital
allocation and risk limits. Forced liquidation at the test endpoint is included. Bar-level fills,
fixed costs and sampled risk limits cannot reproduce order-book liquidity, outages or real execution.
These results do not establish future profitability. This program has no real-order submission code.</p>
<p>Machine-readable details: <a href="training_report.json">training_report.json</a> ·
<a href="test_trades.csv">test_trades.csv</a> · <a href="test_equity.csv">test_equity.csv</a></p>
</main></html>"""
    atomic_bytes(run_dir / "report.html", html.encode())
