#!/usr/bin/env python3
"""
Generates the Milestone 1 figures from the measured output of ./bs_fdm.

Re-run this on YOUR machine after running the solver there, so the numbers in
the report match your own hardware:

    ./bs_fdm converge > converge.txt
    ./bs_fdm bench    > bench.txt
    python3 make_figures.py

(The arrays below are pre-filled with a reference run so the script works
out of the box; replace them with your own numbers.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- measured data: ./bs_fdm converge -------------------------------------
N_conv   = [128, 256, 512, 1024, 2048]
dx       = [1.562e-2, 7.812e-3, 3.906e-3, 1.953e-3, 9.766e-4]
abs_err  = [5.710e-4, 1.455e-4, 3.626e-5, 9.075e-6, 2.269e-6]

# ---- measured data: ./bs_fdm bench ----------------------------------------
N_bench  = [256, 512, 1024, 2048]
time_s   = [0.0001, 0.0008, 0.0057, 0.0294]
updates  = [175950, 1410360, 11291874, 90377097]
gflops   = [9.526, 8.722, 9.962, 15.367]

# ---- measured data: ./bs_fdm 2d -------------------------------------------
N_2d     = [64, 128, 256, 512]
time_2d  = [0.0007, 0.0093, 0.1462, 2.8700]

# ---------------------------------------------------------------------------
# Figure 1 — spatial convergence (log-log, slope should be 2)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(dx, abs_err, "o-", lw=1.8, label="explicit Euler (measured)")
ref = [abs_err[0] * (d / dx[0]) ** 2 for d in dx]
ax.loglog(dx, ref, "k--", lw=1.2, label=r"$O(\Delta x^2)$ reference")
ax.set_xlabel(r"grid spacing $\Delta x$ (log-price)")
ax.set_ylabel("absolute pricing error vs Black–Scholes")
ax.set_title("Spatial convergence of the sequential baseline")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("fig1_convergence.png", dpi=160)

# ---------------------------------------------------------------------------
# Figure 2 — work and runtime growth
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))

a1.loglog(N_bench, updates, "s-", lw=1.8, label="lattice updates")
cubic = [updates[0] * (n / N_bench[0]) ** 3 for n in N_bench]
a1.loglog(N_bench, cubic, "k--", lw=1.2, label=r"$O(N^3)$ reference")
a1.set_xlabel("grid points $N$")
a1.set_ylabel("total lattice-site updates")
a1.set_title("Work grows cubically (stability-limited $\\Delta t$)")
a1.grid(True, which="both", alpha=0.3)
a1.legend()

a2.semilogx(N_bench, gflops, "o-", lw=1.8, color="tab:red")
a2.set_xlabel("grid points $N$")
a2.set_ylabel("GFLOP/s (single thread)")
a2.set_title("Sustained single-thread throughput")
a2.set_ylim(0, max(gflops) * 1.4)
a2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("fig2_baseline_scaling.png", dpi=160)

# ---------------------------------------------------------------------------
# Figure 3 — 2D runtime wall (the case for the GPU)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(N_2d, time_2d, "^-", lw=1.8, color="tab:purple",
          label="2-asset, 9-point stencil (measured)")
quart = [time_2d[0] * (n / N_2d[0]) ** 4 for n in N_2d]
ax.loglog(N_2d, quart, "k--", lw=1.2, label=r"$O(N^4)$ reference")
ax.set_xlabel("grid points per dimension $N$")
ax.set_ylabel("wall-clock time (s)")
ax.set_title("2-asset solver: runtime motivating parallelisation")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("fig3_2d_runtime.png", dpi=160)

print("wrote fig1_convergence.png, fig2_baseline_scaling.png, fig3_2d_runtime.png")
