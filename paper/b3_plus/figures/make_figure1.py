"""Vector method figure from the measured spam replay; figures4papers styling."""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/promising_directions"))
from style import PALETTE as P, apply_style
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from b3_plus_experiments import episode_u
from b3_plus_policy import PlusParams, b1, run_plus_policies

def spam_run():
    """Labels per 100-event bin (mean over seeds) for P3b and P7s, and the mean promotion time."""
    z = np.load(ROOT / "docs/promising_directions/results/b3/losses/spam.npz")
    u = episode_u(0, 100, 2000)
    l0, l1 = (np.repeat(z[k][0].astype(np.int8)[:, None], 100, axis=1) for k in ("l0", "l1"))
    params = b1.Params(T=2000, delay=50, cq=0.05)
    ours = PlusParams(variant="P7s", acq_shape="settled")
    # multiplier 1 in every state is the incumbent pacer
    incumbent = PlusParams(
        variant="P7s", acq_shape="settled", settled_floor=1.0, use_hopeless=False
    )
    bins, promoted = [], None
    for plus in (incumbent, ours):
        out, objs = run_plus_policies(
            l0, l1, u, 200, 5.0, 0.05, params, [plus], trace=True, return_objs=True
        )
        bins.append(objs["P7s"].tr["ask"].mean(axis=1).reshape(20, 100).sum(axis=1))
        promoted = float(np.nanmean(out["P7s"]["t_prom"]))

    records = {}
    path = ROOT / "docs/promising_directions/results/b3_plus/spam__S0__h2000.jsonl"
    for line in path.read_text().splitlines():
        records.update(json.loads(line)["variants"])
    for b, name in zip(bins, ("P3b", "P7s")):
        assert abs(b.sum() - records[name]["mean_labels"]) < 0.05, (name, b.sum())
    return bins, promoted



def main():
    (incumbent, ours), promoted = spam_run()
    apply_style(9, axes_linewidth=1.2)
    fig = plt.figure(figsize=(6.5, 3.8))
    ax = fig.add_axes([.115,.52,.84,.37])
    events = np.arange(21)*100
    for values, label, color, marker, line in (
        (incumbent,"Certified procedure (P3b)",P["red_strong"],"s","--"),
        (ours,"Settled-decision (ours)",P["blue_main"],"o","-"),
    ):
        cumulative = np.r_[0,np.cumsum(values)]
        ax.plot(events,cumulative,label=label,color=color,marker=marker,markevery=4,
                markersize=3.6,lw=1.8,ls=line)
        ax.text(2035,cumulative[-1],f"{cumulative[-1]:.0f}",va="center",fontsize=10,
                color=color,fontweight="bold")
    ax.axvline(promoted,color=P["dark"],lw=1.1,ls=":")
    ax.text(promoted+35,178,f"Promotion\nevent {promoted:.0f}",fontsize=8.5,va="top")
    ax.set_xlim(0,2200);ax.set_ylim(0,220)
    ax.set_xticks([0,500,1000,1500,2000]);ax.set_yticks([0,50,100,150,200])
    ax.set_xlabel("Event");ax.set_ylabel("Cumulative labels")
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,fontsize=8.5,
              borderaxespad=0,handlelength=2.8,columnspacing=1.3)
    fig.text(.025,.96,"(a) Same promotion, lower label spending",fontsize=10,fontweight="bold")

    loop = fig.add_axes([.025,.055,.50,.30]);loop.axis("off")
    loop.text(0,1.09,"(b) Preserve the decision procedure",fontsize=9,fontweight="bold")
    for x,text in ((.0,"Issue\nquery + bet"),(.355,"Release\nwait d events"),(.71,"Decide\npromote / rollback")):
        loop.add_patch(FancyBboxPatch((x,.36),.28,.44,boxstyle="round,pad=.015,rounding_size=.03",
                                     fc="white",ec=P["dark"],lw=1.0))
        loop.text(x+.14,.58,text,ha="center",va="center",fontsize=8)
    for left in (.29,.645):
        loop.annotate("",xy=(left+.055,.58),xytext=(left,.58),
                      arrowprops=dict(arrowstyle="->",lw=1.1,color=P["dark"]))
    loop.text(.0,.13,"Repeat at each event; act after released feedback.",fontsize=8)

    states = fig.add_axes([.595,.055,.38,.30]);states.axis("off")
    states.text(0,1.09,"(c) Attenuate acquisition",fontsize=9,fontweight="bold")
    for y,state,rate in ((.73,"Settled","0 × pace"),(.48,"Hopeless","¼ × pace"),(.23,"Otherwise","1 × pace")):
        states.text(0,y,state,fontsize=8.5,va="center")
        states.text(.98,y,rate,fontsize=8.5,ha="right",va="center",fontweight="bold")
    states.text(0,-.02,"Positive query floor in every state",fontsize=8)
    fig.savefig(HERE / "fig1_method.pdf",facecolor="white")
    fig.savefig(HERE / "fig1_method.png",dpi=300,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
