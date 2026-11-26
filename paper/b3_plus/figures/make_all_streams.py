"""Rebuild manuscript numbers, tables, figures and report from the 20 configured streams.

Run: python paper/b3_plus/figures/make_all_streams.py
Every policy mean first averages query seeds within an episode, then episodes within a
stream. Cross-stream means and medians give each stream equal weight.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = HERE.parent / "tables"
sys.path.insert(0, str(ROOT / "scripts/promising_directions"))
from real_datasets import OUT, load_results, validate_losses
from style import PALETTE as P, apply_style
import matplotlib.pyplot as plt

FAMILIES = ("Electricity meters", "Air quality", "Demand and mobility", "Other real streams")
VARIANTS = ("P0", "P3b", "P6", "P7s")
TOL = 1e-9


def family(name):
    if name.startswith("elec_MT_"):
        return FAMILIES[0]
    if name.startswith("air_"):
        return FAMILIES[1]
    if name in ("metro", "bike_hour"):
        return FAMILIES[2]
    return FAMILIES[3]


def load():
    rows = []
    for i, (name, result) in enumerate(load_results().items()):
        eps = result["episodes"]
        row = {"name": name, "family": family(name), "K": result["K"],
               "delay": result["delay"], "same": sum(e["same_deployment"] for e in eps)}
        for variant in VARIANTS:
            for field, suffix in (("mean_cost", ""), ("mean_labels", "_lab"),
                                  ("mean_loss", "_loss"), ("mean_switches", "_switch")):
                row[variant + suffix] = float(np.mean([e[variant][field] for e in eps]))
        row["diff"] = row["P7s"] - row["P3b"]
        row["cost_saving"] = -100 * row["diff"] / row["P3b"]
        row["saving"] = 1 - row["P7s_lab"] / row["P3b_lab"]
        row["promotes"] = bool(np.mean([e["P3b"]["promotion_rate"] for e in eps]) > 0.5)
        row["loss_diff"] = row["P7s_loss"] - row["P3b_loss"]
        row["query_diff"] = .05 * (row["P7s_lab"] - row["P3b_lab"])
        row["switch_diff"] = 5 * (row["P7s_switch"] - row["P3b_switch"])
        # The compact replay records retain episode means, not seed-level cost arrays.
        row["ci"] = None
        if row["K"] >= 5:
            d = np.array([e["P7s"]["mean_cost"] - e["P3b"]["mean_cost"] for e in eps])
            rng = np.random.default_rng(np.random.SeedSequence([20261108, i]))
            boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
            row["ci"] = np.percentile(boots, [2.5, 97.5]).tolist()
        rows.append(row)
    return rows


def counts(rows):
    d = np.array([r["diff"] for r in rows])
    return int((d < -TOL).sum()), int((abs(d) <= TOL).sum()), int((d > TOL).sum())


def tex_name(name):
    return name.replace("_", r"\_")


def write_tables(rows):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Family & Streams & Episodes & Median labels saved & Cheaper / equal / dearer & Matching summaries & P6 cheaper \\", r"\midrule"]
    for fam in (*FAMILIES, "All"):
        group = [r for r in rows if fam in ("All", r["family"])]
        win, tie, loss = counts(group)
        same = sum(r["same"] for r in group) / sum(r["K"] for r in group)
        p6 = sum(r["P6"] < r["P7s"] - TOL for r in group)
        if fam == "All": lines.append(r"\midrule")
        lines.append(f"{fam} & {len(group)} & {sum(r['K'] for r in group)} & "
                     f"{100*np.median([r['saving'] for r in group]):.0f}\\% & {win} / {tie} / {loss} & "
                     f"{100*same:.0f}\\% & {p6} \\\\")
    (TABLES / "table4_families.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Stream & $K$ & $P_0$ & P3b & P6 & Ours & Cost saved & Labels P3b$\to$ours \\", r"\midrule"]
    for r in rows:
        lines.append(f"{tex_name(r['name'])} & {r['K']} & {r['P0']:.2f} & {r['P3b']:.2f} & "
                     f"{r['P6']:.2f} & {r['P7s']:.2f} & {r['cost_saving']:+.2f}\\% & "
                     f"{r['P3b_lab']:.0f}$\\to${r['P7s_lab']:.0f} \\\\")
    (TABLES / "table1_main.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    head = r"Stream & $d$ & $\Delta$ cost & 95\% episode interval & $\Delta$ loss & $\Delta$ query cost & $\Delta$ switch cost \\"
    lines = [r"\begin{longtable}{lrrrrrr}",
             r"\caption{Per-stream differences (ours minus P3b). Intervals use 10,000 paired episode-bootstrap resamples; dashes indicate fewer than five episodes.}\label{tab:streams}\\",
             r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head,
             r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    for r in rows:
        ci = (f"$[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]$" if r["ci"] else r"\textemdash")
        lines.append(f"{tex_name(r['name'])} & {r['delay']} & ${r['diff']:+.2f}$ & {ci} & "
                     f"${r['loss_diff']:+.2f}$ & ${r['query_diff']:+.2f}$ & ${r['switch_diff']:+.2f}$ \\\\")
    (TABLES / "table5_streams.tex").write_text("\n".join(lines + [r"\end{longtable}"]) + "\n")


def write_numbers(rows):
    win, tie, loss = counts(rows)
    promoted = [r for r in rows if r["promotes"]]
    other = [r for r in rows if not r["promotes"]]
    median_saving = lambda group: f"{100*np.median([r['saving'] for r in group]):.0f}" if group else "0"
    numbers = {
        "NStreams": len(rows), "NEpisodes": sum(r["K"] for r in rows),
        "NCheaper": win, "NEqual": tie, "NDearer": loss,
        "MaxDearer": f"{max(r['diff'] for r in rows):.2f}",
        "MeanSaving": f"{100*np.mean([r['saving'] for r in rows]):.0f}",
        "MedSaving": median_saving(rows), "MaxSaving": f"{100*max(r['saving'] for r in rows):.0f}",
        "NPromoted": len(promoted), "MedSavingPromoted": median_saving(promoted),
        "NOther": len(other), "MedSavingOther": median_saving(other),
        "NSameEpisodes": sum(r["same"] for r in rows),
        "NSameStreams": sum(r["same"] == r["K"] for r in rows),
        "MedCostChange": f"{np.median([r['cost_saving'] for r in rows]):.2f}",
        "MeanCostChange": f"{np.mean([r['cost_saving'] for r in rows]):.2f}",
        "NSixCheaper": sum(r["P6"] < r["P7s"] - TOL for r in rows),
        "NBeatNever": sum(r["P7s"] < r["P0"] - TOL for r in rows),
        "NIncBeatNever": sum(r["P3b"] < r["P0"] - TOL for r in rows),
        "NLowPower": sum(r["K"] < 5 for r in rows),
    }
    (TABLES / "numbers_all.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in numbers.items()))
    (TABLES / "table0_summary.tex").write_text(
        "\n".join([
            r"\begin{tabular}{lrr}", r"\toprule",
            r"Metric relative to P3b & Mean & Median \\", r"\midrule",
            f"Net cost saved & {numbers['MeanCostChange']}\\% & {numbers['MedCostChange']}\\% \\\\",
            f"Labels saved & {numbers['MeanSaving']}\\% & {numbers['MedSaving']}\\% \\\\",
            r"\bottomrule", r"\end{tabular}", ""
        ])
    )
    return numbers


def write_details():
    """Retained ablation and sensitivity measurements; no imputation for unrun settings."""
    base = OUT.parent
    def records(path):
        episodes = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            episodes.setdefault(row["episode"], {}).update(row["variants"])
        return episodes
    cases = ("spam", "rialto")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{spam ($K=1$)} & \multicolumn{2}{c}{rialto ($K=16$)} \\",
             r"Variant & Cost & $\Delta$ & Cost & $\Delta$ \\", r"\midrule"]
    variants = (("P3b", "Incumbent certified switching"), ("P4ref", "Cost-aware promotion"),
                ("P7b", r"Acquisition: \textsc{hopeless} only"),
                ("P7s", r"Acquisition: both states"))
    eps = {n: records(base / "ablation" / f"{n}__S0__h2000.jsonl") for n in cases}
    for variant, label in variants:
        vals = []
        for n in cases:
            cost = np.mean([e[variant]["mean_cost"] for e in eps[n].values()])
            ref = np.mean([e["P3b"]["mean_cost"] for e in eps[n].values()])
            vals.extend([f"{cost:.2f}", f"${cost-ref:+.2f}$"])
        lines.append(label + " & " + " & ".join(vals) + r" \\")
    (TABLES / "table3_ablation.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    sensitivity = {}
    for n in (*cases, "openml_ldpa"):
        sensitivity[n] = {}
        for c in ("S0", "S1", "S2", "S3", "S4"):
            path = base / f"{n}__{c}__h2000.jsonl"
            if path.exists():
                ee = records(path)
                if all("P7s" in e for e in ee.values()):
                    sensitivity[n][c] = float(np.mean([e["P7s"]["mean_cost"] - e["P3b"]["mean_cost"] for e in ee.values()]))
    # Additional retained case measurements carry their original configuration labels.
    source = base / "figure_data/fig4_sensitivity.csv"
    with source.open() as f:
        for r in csv.DictReader(f):
            if r["case"] == "openml_ldpa":
                sensitivity[r["case"]][r["config"]] = float(r["P7s_minus_P3b"])
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Stream & S0 & S1 ($B=400$) & S2 ($c_s=20$) & S3 ($2\times d$) & S4 ($c_q=0.10$) \\", r"\midrule"]
    for n, configs in sensitivity.items():
        vals = [f"${configs[c]:+.2f}$" if c in configs else r"\textemdash" for c in ("S0","S1","S2","S3","S4")]
        lines.append(tex_name(n) + " & " + " & ".join(vals) + r" \\")
    (TABLES / "table2_sensitivity.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    return sensitivity


def save_figure(fig, name):
    """Vector paper asset and 300-dpi preview, following figures4papers conventions."""
    fig.savefig(HERE / f"{name}.pdf", facecolor="white")
    fig.savefig(HERE / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def draw(rows):
    apply_style(9, axes_linewidth=1.2)
    blue, dark, grey = P["blue_main"], P["dark"], P["grey"]
    for key, xlabel, name in (("cost_saving", "Net cost saved relative to P3b (%)", "fig2_netcost"),
                              ("saving", "Labels saved relative to P3b (%)", "fig3_labels")):
        fig, ax = plt.subplots(figsize=(6.5, 4.7))
        values = np.array([r[key] * (100 if key == "saving" else 1) for r in rows])
        mean, median = float(values.mean()), float(np.median(values))
        bars = ax.barh(range(len(rows)), values, height=.7,
                       color=[blue if v >= 0 else P["red_strong"] for v in values],
                       edgecolor=dark, linewidth=.65, zorder=3)
        for bar, value in zip(bars, values):
            if value < 0:
                bar.set_hatch("///")
        ax.set_yticks(range(len(rows)), [r["name"] for r in rows], fontsize=8.5)
        ax.invert_yaxis()
        ax.axvline(0, color=dark, lw=1.1, zorder=2)
        ax.axvline(mean, color=dark, lw=1.35, ls="--", zorder=4,
                   label=f"Mean {mean:.2f}%" if key == "cost_saving" else f"Mean {mean:.0f}%")
        ax.axvline(median, color=grey, lw=1.5, ls=":", zorder=4,
                   label=f"Median {median:.2f}%" if key == "cost_saving" else f"Median {median:.0f}%")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2, fontsize=9,
                  borderaxespad=0, handlelength=2.4, columnspacing=1.8)
        ax.set_xlabel(xlabel, labelpad=7)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#E6E6E6", lw=.5)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(rows)-.4, -.7)
        ax.set_xlim(-.45 if key == "cost_saving" else 0, 6.05 if key == "cost_saving" else 86)
        for i, value in enumerate(values):
            label = f"{value:.2f}" if key == "cost_saving" else f"{value:.0f}"
            ax.text(max(0, value) + (.07 if key == "cost_saving" else 1.2), i, label,
                    va="center", fontsize=8.5, color=dark, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.15))
        fig.subplots_adjust(left=.245, right=.985, bottom=.12, top=.9)
        save_figure(fig, name)

    # Headline aggregates and the full dataset distribution share the same zero baseline.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    for ax, key, title, label_format in (
        (axes[0], "cost_saving", "(a) Net cost saved", ".2f"),
        (axes[1], "saving", "(b) Labels saved", ".0f"),
    ):
        values = np.array([r[key] * (100 if key == "saving" else 1) for r in rows])
        mean, median = float(values.mean()), float(np.median(values))
        bars = ax.bar([0, 1], [mean, median], width=.6,
                      color=[blue, P["neutral"]], edgecolor=dark, linewidth=1.0, zorder=3)
        for bar, value in zip(bars, (mean, median)):
            ax.annotate(f"{value:{label_format}}%", (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
        ax.scatter(2.25 + np.linspace(-.14,.14,len(values)), values, s=22,
                   facecolors="white", edgecolors=dark, linewidths=.8, zorder=4)
        ax.axhline(0, color=dark, lw=.9)
        ax.set_xticks([0,1,2.25], ["Mean", "Median", "20 datasets"])
        ax.set_ylabel("Saving relative to P3b (%)")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=10)
        ax.set_xlim(-.6,2.7)
        ax.set_ylim(-.35 if key == "cost_saving" else 0,
                    5.9 if key == "cost_saving" else 85)
        ax.yaxis.grid(True, color="#E6E6E6", lw=.5, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.1, w_pad=1.8)
    save_figure(fig, "fig4_all_streams")


def write_report(rows,numbers,sensitivity):
    summary={"configuration":{"T":2000,"B":200,"cq":.05,"cs":5,"query_seeds":100},
             "aggregation":"equal stream weight; episode means average the 100 paired query seeds",
             "numbers":numbers,"streams":rows,"sensitivity":sensitivity}
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2)+'\n')
    fields=["name","K","delay","P0","P3b","P6","P7s","cost_saving","P3b_lab","P7s_lab","saving","same"]
    with (OUT / "summary.csv").open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    text=["# Real-data results\n", f"The evaluation contains {len(rows)} real datasets and {numbers['NEpisodes']} episodes. "
          "Each episode has 2,000 events and 100 paired query seeds; B=200, query cost=0.05, switch cost=5.\n",
          f"Ours is cheaper than P3b on {numbers['NCheaper']} streams, equal on {numbers['NEqual']}, "
          f"and dearer on {numbers['NDearer']}. Mean net-cost saving: {numbers['MeanCostChange']}%; "
          f"mean labels saved: {numbers['MeanSaving']}%. The corresponding medians are "
          f"{numbers['MedCostChange']}% and {numbers['MedSaving']}%. "
          f"P6 is cheaper than ours on {numbers['NSixCheaper']} streams.\n",
          f"Deployment loss, switch count and rollback count match P3b on every query seed in "
          f"{numbers['NSameEpisodes']} of {numbers['NEpisodes']} episodes. Equal totals do not imply identical action trajectories.\n",
          "Stream summaries weight episodes equally; cohort medians weight streams equally. Bootstrap intervals "
          "use 10,000 paired episode resamples for streams with at least five episodes; they are omitted "
          "for shorter streams. Related meters and stations are correlated.\n",
          "| Stream | Episodes | Net cost saved | Labels | Labels saved |\n|---|---:|---:|---:|---:|"]
    for r in rows:text.append(f"| {r['name']} | {r['K']} | {r['cost_saving']:+.2f}% | {r['P3b_lab']:.0f} → {r['P7s_lab']:.0f} | {100*r['saving']:.0f}% |")
    text += ["\nRegenerate with `python paper/b3_plus/figures/make_all_streams.py`. "
             "The authoritative scope is `configs/promising_directions/real_datasets.json`; frozen loss "
             "files and their SHA-256 hashes are under `results/b3_plus/all_streams/`. "
             "Ablation measurements cover spam and rialto; sensitivity measurements cover these and openml_ldpa."]
    (ROOT / "docs/promising_directions/B3_PLUS_RESULTS.md").write_text('\n'.join(text)+'\n')


if __name__ == "__main__":
    validate_losses()
    data=load()
    write_tables(data)
    numbers=write_numbers(data)
    sensitivity=write_details()
    draw(data)
    write_report(data,numbers,sensitivity)
    print(numbers)
