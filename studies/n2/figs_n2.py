"""Figures for the N2 manuscript. fig1_media: chi(2) magnitude and phase, pump absorption and coherence length against
bias (ADQW, TW; 5e11 cm^-2, design pump energy). fig2_anatomy: programmed pattern against the best two-field pattern
(ADQW, 1.26 W, realistic fields): field profile, in-phase coupling, powers. fig3_tuning (sat_tuning.json) and
fig4_electrodes (sat_electrodes.json) when those exist. Writes ../../paper_n2/figures/*.pdf."""
import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import common_n2 as c

outdir = os.path.join(here, "..", "..", "paper_n2", "figures"); os.makedirs(outdir, exist_ok=True)
BLUE, ORANGE, AQUA, PURPLE, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a4fbf", "#c4314b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.6, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.3, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
J = lambda f: json.load(open(os.path.join(here, f)))


def clean(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def label(ax, s, x=-0.2):
    ax.text(x, 1.03, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")


def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.45))
    for d, col, lab in (("ad", BLUE, "ADQW"), ("tw", ORANGE, "TW")):
        b = c.Bank(d, 4, 0.0)
        a1 = np.array([m.alpha(m.rho0)[0] for m in b.media[:len(b.F)]]) / 100 * b.wg["G1"]
        axs[0].plot(b.F, np.abs(b.chi0) * 1e9, color=col, label=lab)
        axs[1].plot(b.F, np.degrees(np.angle(b.chi0)), color=col, marker="o", ms=1.6, lw=0.0 if d == "tw" else 1.0, label=lab)
        axs[2].plot(b.F, a1, color=col, label=lab)
    axs[0].set_ylabel(r"$|\chi^{(2)}|$ (nm/V)"); axs[1].set_ylabel(r"arg $\chi^{(2)}$ (deg)"); axs[2].set_ylabel(r"modal pump absorption (cm$^{-1}$)")
    axs[2].set_yscale("log")
    for k, ax in enumerate(axs):
        ax.set_xlabel("bias $F$ (kV/cm)"); clean(ax); label(ax, "abc"[k])
    axs[0].legend(frameon=False)
    fig.savefig(os.path.join(outdir, "fig1_media.pdf")); plt.close(fig)


def profile(bank, F_seg, L_seg, P, dx=0.25, gap=c.GAP_UM):
    idx = c.realistic_steps(bank, F_seg, L_seg, dx, gap_um=gap)
    r = c.propagate(bank, idx, dx, P, record=True)
    Fy = c.realistic_fields(F_seg, np.broadcast_to(np.asarray(L_seg, float), (len(F_seg),)), dx, len(idx) * dx, gap)
    area = bank.wg["t_stack_um"] * 1e-6 * bank.wg["w_um"] * 1e-6
    a = r["amp"]; th = r["theta_x"]
    inph = []
    for k, i in enumerate(idx):
        m = bank.media[i]
        rho = m.populations(abs(a[k, 0]) ** 2 * bank.wg["G1"] / area, abs(a[k, 1]) ** 2 * bank.wg["G2"] / area)
        cc = -1j * m.chi(rho) * a[k, 0] ** 2 * np.exp(1j * th[k])
        ref = a[k, 1] if abs(a[k, 1]) > 0 else cc
        inph.append(np.real(cc * np.conj(ref)) / abs(ref) / max(abs(a[k, 0]) ** 2, 1e-30))
    return r, Fy[1], np.array(inph)


def fig2():
    M = {(x["design"], x["P"]): x for x in J("sat_main_I4.json")}
    r = M[("ad", 1.26)]
    b = c.Bank("ad", 4, 0.0)
    pat = r["programmed"]["realistic"]["pattern"]; pitch = r["pitch"]
    Fa, Fb = [int(v) for v in re.findall(r"[+-]\d+", r["twofield"]["realistic"]["label"])]
    sa, sb = r["twofield"]["realistic"]["scales"]
    la, lb = sa * b.Lcoh[b.index_of(Fa)], sb * b.Lcoh[b.index_of(Fb)]
    n = int(np.ceil(r["twofield"]["realistic"]["L"] / (la + lb))) + 1
    rp, Fp, ip = profile(b, pat, pitch, 1.26, gap=r["gap"])
    rt, Ft, it = profile(b, [Fa, Fb] * n, [la, lb] * n, 1.26, gap=r["gap"])
    fig, axs = plt.subplots(3, 1, figsize=(5.0, 5.0), sharex=True, gridspec_kw=dict(hspace=0.18))
    x = (np.arange(len(Fp)) + 0.5) * 0.25; xt = (np.arange(len(Ft)) + 0.5) * 0.25
    axs[0].plot(xt, Ft, color=INK2, lw=1.0, label=f"two-field ({Fa:+d}, {Fb:+d}) kV/cm")
    axs[0].plot(x, Fp, color=BLUE, label="programmed")
    axs[0].set_ylabel("field at stack centre\n(kV/cm)"); axs[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    axs[1].plot(rt["x_um"][:-1], it * 1e9, color=INK2, lw=1.0)
    axs[1].plot(rp["x_um"][:-1], ip * 1e9, color=BLUE)
    axs[1].axhline(0, color=INK2, lw=0.5)
    axs[1].set_ylabel(r"in-phase $\chi^{(2)}$" + "\n(nm/V)")
    for rr, col in ((rt, INK2), (rp, BLUE)):
        axs[2].plot(rr["x_um"], 100 * rr["P2"] / 1.26, color=col)
        axs[2].plot(rr["x_um"], rr["P1"] / 1.26, color=col, ls="--", lw=0.9)
    axs[2].set_ylabel(r"$P_2/P_1(0)$ (%), solid" + "\n" + r"$P_1/P_1(0)$, dashed"); axs[2].set_xlabel(r"$x$ ($\mu$m)")
    axs[2].set_xlim(0, max(rp["L_opt_um"], rt["L_opt_um"]) + 20)
    for k, ax in enumerate(axs):
        clean(ax); label(ax, "abc"[k], x=-0.13)
    fig.savefig(os.path.join(outdir, "fig2_anatomy.pdf")); plt.close(fig)


def fig3():
    T = J("sat_tuning.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.3))
    for ax, d, lab in ((axs[0], "ad", "ADQW"), (axs[1], "tw", "TW")):
        rr = sorted([x for x in T if x["design"] == d], key=lambda x: x["dhw"])
        x = [1e3 * r["dhw"] for r in rr]
        ax.plot(x, [100 * r["fixed_design_fields"] for r in rr], color=INK2, ls="--", marker="o", ms=2.4, lw=1.0, label="fixed device, design fields")
        ax.plot(x, [100 * max(r["fixed_retuned"]["eta"], r["fixed_design_fields"]) for r in rr], color=INK2, marker="o", ms=2.4, label="fixed device, both fields retuned")
        ax.plot(x, [100 * r["uniform"] for r in rr], color=AQUA, marker="s", ms=2.4, lw=1.0, label="uniform bias")
        ax.plot(x, [100 * r["programmed_fixed"] for r in rr], color=BLUE, ls="--", marker="o", ms=2.4, lw=1.0, label="programmed, design pattern")
        ax.plot(x, [100 * r["reprogrammed"]["eta"] for r in rr], color=BLUE, marker="o", ms=2.8, label="reprogrammed")
        ax.set_yscale("log"); ax.set_yticks([t for t in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if t >= (0.01 if d == "ad" else 0.0)])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("pump detuning from design (meV)"); ax.set_ylabel(r"$\eta$ (%) at 1.26 W")
        ax.set_title(lab, fontsize=8.5); clean(ax)
    axs[0].legend(frameon=False, fontsize=5.8, loc="lower right"); label(axs[0], "a", x=-0.18); label(axs[1], "b", x=-0.18)
    fig.savefig(os.path.join(outdir, "fig3_tuning.pdf")); plt.close(fig)


def fig4():
    E = J("sat_electrodes.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw=dict(wspace=0.35))
    rr = sorted([x for x in E if x["K"] == "all" and x["limit"] and not (x["pitch"] >= 4 and x["gap"] < 2)], key=lambda x: x["pitch"])
    bench = 100 * rr[0]["benchmark"]
    axs[0].plot([r["pitch"] for r in rr], [100 * r["eta"] for r in rr], color=BLUE, marker="o", ms=3, label="programmed")
    nl = [x for x in E if x["K"] == "all" and not x["limit"]]
    if nl:
        axs[0].plot([nl[0]["pitch"]], [100 * nl[0]["eta"]], color=ORANGE, marker="^", ms=4, lw=0, label="no breakdown limit")
    g1 = [x for x in E if x["K"] == "all" and x["limit"] and x["pitch"] == 4.0 and x["gap"] < 2]
    if g1:
        axs[0].plot([g1[0]["pitch"]], [100 * g1[0]["eta"]], color=AQUA, marker="s", ms=3.5, lw=0, label=r"1 $\mu$m gaps")
    axs[0].axhline(bench, color=INK2, ls=":", lw=1.0, label="best two-field pattern")
    axs[0].set_xscale("log", base=2); axs[0].set_xticks([2, 4, 8, 16]); axs[0].set_xticklabels(["2", "4", "8", "16"])
    axs[0].minorticks_off()
    axs[0].set_xlabel(r"electrode pitch ($\mu$m)"); axs[0].set_ylabel(r"$\eta$ (%) at 1.26 W, ADQW"); axs[0].legend(frameon=False, fontsize=5.8)
    rr = sorted([x for x in E if x["pitch"] == 4.0 and x["limit"] and x["gap"] == 2.0], key=lambda x: x["n_levels"])
    axs[1].plot([r["n_levels"] for r in rr], [100 * r["eta"] for r in rr], color=BLUE, marker="o", ms=3)
    axs[1].axhline(bench, color=INK2, ls=":", lw=1.0)
    axs[1].set_xscale("log", base=2); axs[1].set_xticks([r["n_levels"] for r in rr]); axs[1].set_xticklabels([str(r["n_levels"]) for r in rr])
    axs[1].minorticks_off(); axs[1].set_xlabel("number of voltage levels (4 $\\mu$m pitch)"); axs[1].set_ylabel(r"$\eta$ (%) at 1.26 W, ADQW")
    for k, ax in enumerate(axs):
        clean(ax); label(ax, "ab"[k], x=-0.2)
    fig.savefig(os.path.join(outdir, "fig4_electrodes.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2()
    for f, fn in (("sat_tuning.json", "fig3"), ("sat_electrodes.json", "fig4")):
        if os.path.exists(os.path.join(here, f)) and fn in globals():
            globals()[fn]()
    print("figures written to", os.path.abspath(outdir))
