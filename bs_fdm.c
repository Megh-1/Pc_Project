/* ===========================================================================
 * bs_fdm.c — Sequential baseline solver for the Black–Scholes PDE
 *            via Finite Difference Methods (FDM).
 *
 * Milestone 1 deliverable for:
 *   "Parallel Multi-Asset Options Pricing Engine (Black-Scholes PDE Stencil)"
 *
 * Contents
 *   1. Analytical references  : Black–Scholes closed form (European),
 *                               Margrabe closed form (2-asset exchange option)
 *   2. 1D solver             : log-space explicit Euler  (3-point stencil)
 *                               log-space Crank–Nicolson (tridiagonal / Thomas)
 *                               American early exercise via projection
 *   3. 2D solver             : two correlated assets, 9-point stencil
 *                               (cross-derivative term from correlation)
 *   4. Validation harness    : L-infinity / L2 error vs analytical,
 *                               spatial convergence order estimate
 *   5. Benchmark harness     : wall time, MLUP/s, GFLOP/s, arithmetic intensity
 *
 * Build:
 *   gcc -O3 -march=native -std=c11 bs_fdm.c -o bs_fdm -lm
 *
 * Usage:
 *   ./bs_fdm price      # single European call, explicit + CN, vs analytical
 *   ./bs_fdm american   # American put, early-exercise premium
 *   ./bs_fdm converge   # spatial convergence study (shows order ~2)
 *   ./bs_fdm bench      # latency / throughput baseline table
 *   ./bs_fdm 2d         # 2-asset exchange option vs Margrabe closed form
 *
 * NOTE FOR MILESTONE 2: every hot loop in this file is marked with a
 * "PARALLEL HOTSPOT" comment. Those are the loops that get OpenMP pragmas
 * and become CUDA kernels. Nothing else needs to change.
 * ===========================================================================*/

#define _POSIX_C_SOURCE 199309L   /* clock_gettime under -std=c11 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

/* --------------------------------------------------------------------------
 * 0. Small utilities
 * --------------------------------------------------------------------------*/

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double norm_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }
static double norm_pdf(double x) { return exp(-0.5 * x * x) / sqrt(2.0 * M_PI); }

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

/* --------------------------------------------------------------------------
 * 1. Analytical references (ground truth for accuracy metrics)
 * --------------------------------------------------------------------------*/

typedef struct {
    double price, delta, gamma;
} greeks_t;

/* European vanilla under Black–Scholes–Merton (continuous dividend yield q). */
static greeks_t bs_analytic(double S, double K, double r, double q,
                            double sigma, double T, int is_call)
{
    greeks_t g;
    double srt = sigma * sqrt(T);
    double d1  = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / srt;
    double d2  = d1 - srt;
    double dfq = exp(-q * T), dfr = exp(-r * T);

    if (is_call)
        g.price = S * dfq * norm_cdf(d1) - K * dfr * norm_cdf(d2);
    else
        g.price = K * dfr * norm_cdf(-d2) - S * dfq * norm_cdf(-d1);

    g.delta = is_call ? dfq * norm_cdf(d1) : -dfq * norm_cdf(-d1);
    g.gamma = dfq * norm_pdf(d1) / (S * srt);
    return g;
}

/* Margrabe (1978) exchange option: payoff max(S1 - S2, 0).
 * Used as the closed-form check for the 2D correlated-asset solver, because
 * a true basket option has no closed form. */
static double margrabe(double S1, double S2, double q1, double q2,
                       double s1, double s2, double rho, double T)
{
    double s   = sqrt(s1 * s1 + s2 * s2 - 2.0 * rho * s1 * s2);
    double srt = s * sqrt(T);
    double d1  = (log(S1 / S2) + (q2 - q1 + 0.5 * s * s) * T) / srt;
    double d2  = d1 - srt;
    return S1 * exp(-q1 * T) * norm_cdf(d1) - S2 * exp(-q2 * T) * norm_cdf(d2);
}

/* --------------------------------------------------------------------------
 * 2. Problem definition
 * --------------------------------------------------------------------------*/

typedef struct {
    double S0, K, r, q, sigma, T;
    int    is_call;      /* 1 = call, 0 = put                                */
    int    is_american;  /* 1 = apply early-exercise projection each step    */
    int    N;            /* spatial intervals; grid has N+1 nodes            */
    int    M;            /* time steps; 0 => auto (stability-limited)        */
    double L;            /* half-width of the log-price domain, in sigma*sqrt(T) units */
} params_t;

static params_t default_params(void)
{
    params_t p;
    p.S0 = 100.0; p.K = 100.0; p.r = 0.05; p.q = 0.0;
    p.sigma = 0.20; p.T = 1.0;
    p.is_call = 1; p.is_american = 0;
    p.N = 1024; p.M = 0; p.L = 5.0;
    return p;
}

/* Payoff at maturity. */
static double payoff(double S, double K, int is_call)
{
    double v = is_call ? (S - K) : (K - S);
    return v > 0.0 ? v : 0.0;
}

/* --------------------------------------------------------------------------
 * 3. 1D solver — log-price coordinates
 *
 *   Black–Scholes:   V_t + 1/2 s^2 S^2 V_SS + (r-q) S V_S - r V = 0
 *   tau = T - t   =>  V_tau = 1/2 s^2 S^2 V_SS + (r-q) S V_S - r V
 *   x = ln S      =>  V_tau = 1/2 s^2 V_xx + (r - q - 1/2 s^2) V_x - r V
 *
 * The log substitution makes all coefficients CONSTANT in space. That turns
 * the problem into a constant-coefficient convection-diffusion equation, i.e.
 * exactly the heat-equation stencil the project brief calls for: identical
 * (a, b, c) weights at every interior node, no per-node coefficient loads.
 * This is what makes the kernel bandwidth-bound rather than compute-bound.
 * --------------------------------------------------------------------------*/

/* PARALLEL HOTSPOT #1 — explicit-Euler stencil sweep.
 * Milestone 2: this loop takes  #pragma omp parallel for simd schedule(static)
 * and becomes a 1-D CUDA kernel with one thread per grid node. */
static void step_explicit_1d(const double *restrict Vin, double *restrict Vout,
                             int N, double a, double b, double c)
{
    for (int i = 1; i < N; ++i)
        Vout[i] = a * Vin[i - 1] + b * Vin[i] + c * Vin[i + 1];
}

/* Dirichlet boundaries = the "ghost cells representing extreme market
 * conditions" from the brief.
 *   S -> 0    : a call is worthless; a put is worth the discounted strike.
 *   S -> inf  : a call behaves like a forward; a put is worthless. */
static void apply_bc_1d(double *V, int N, const double *S, double tau,
                        const params_t *p)
{
    double dfr = exp(-p->r * tau), dfq = exp(-p->q * tau);
    if (p->is_call) {
        V[0] = 0.0;
        V[N] = S[N] * dfq - p->K * dfr;
        if (V[N] < 0.0) V[N] = 0.0;
    } else {
        V[0] = p->K * dfr - S[0] * dfq;
        if (V[0] < 0.0) V[0] = 0.0;
        V[N] = 0.0;
    }
    if (p->is_american) {                 /* American value >= intrinsic */
        double p0 = payoff(S[0], p->K, p->is_call);
        double pN = payoff(S[N], p->K, p->is_call);
        if (V[0] < p0) V[0] = p0;
        if (V[N] < pN) V[N] = pN;
    }
}

/* Stability limit for the explicit scheme:
 *   dt <= 1 / (sigma^2/dx^2 + r)
 * The dx^2 term dominates, so refining the grid by 2x costs 4x more time
 * steps => 8x total work. That superlinear growth is precisely why this
 * problem needs a GPU. */
static int auto_timesteps(const params_t *p, double dx)
{
    double dt_max = 1.0 / (p->sigma * p->sigma / (dx * dx) + p->r);
    int    M = (int)ceil(p->T / (0.95 * dt_max));
    return M < 8 ? 8 : M;
}

typedef struct {
    double price, delta, gamma;
    double seconds;
    long long lattice_updates;
} result_t;

/* Explicit Euler. Returns price and Greeks interpolated at S0.
 * Because the domain is centred on ln(S0) and N is even, S0 lands exactly
 * on node N/2 — no interpolation error contaminates the accuracy metric. */
static result_t solve_explicit_1d(const params_t *p_in)
{
    params_t p = *p_in;
    if (p.N % 2) p.N += 1;                       /* keep S0 on a node */

    int    N    = p.N;
    double half = p.L * p.sigma * sqrt(p.T);
    double xmin = log(p.S0) - half, xmax = log(p.S0) + half;
    double dx   = (xmax - xmin) / N;
    int    M    = p.M > 0 ? p.M : auto_timesteps(&p, dx);
    double dt   = p.T / M;

    double *S     = xmalloc((size_t)(N + 1) * sizeof(double));
    double *Vcur  = xmalloc((size_t)(N + 1) * sizeof(double));
    double *Vnext = xmalloc((size_t)(N + 1) * sizeof(double));
    double *intr  = xmalloc((size_t)(N + 1) * sizeof(double));

    for (int i = 0; i <= N; ++i) {
        S[i]    = exp(xmin + i * dx);
        intr[i] = payoff(S[i], p.K, p.is_call);
        Vcur[i] = intr[i];                        /* terminal condition */
    }

    /* Constant stencil weights. */
    double diff = p.sigma * p.sigma / (2.0 * dx * dx);
    double conv = (p.r - p.q - 0.5 * p.sigma * p.sigma) / (2.0 * dx);
    double a = dt * (diff - conv);
    double b = 1.0 - dt * (2.0 * diff + p.r);
    double c = dt * (diff + conv);

    if (b < 0.0)
        fprintf(stderr, "[warn] explicit scheme unstable: b=%g (reduce dt)\n", b);

    double t0 = now_sec();
    for (int n = 0; n < M; ++n) {
        double tau = (n + 1) * dt;
        step_explicit_1d(Vcur, Vnext, N, a, b, c);
        apply_bc_1d(Vnext, N, S, tau, &p);

        if (p.is_american) {
            /* PARALLEL HOTSPOT #2 — pointwise projection, trivially parallel. */
            for (int i = 1; i < N; ++i)
                if (Vnext[i] < intr[i]) Vnext[i] = intr[i];
        }
        double *tmp = Vcur; Vcur = Vnext; Vnext = tmp;
    }
    double t1 = now_sec();

    /* Greeks by finite difference on the log grid:
     *   Delta = (1/S) V_x ,  Gamma = (1/S^2) (V_xx - V_x)              */
    int i0 = N / 2;
    double vx  = (Vcur[i0 + 1] - Vcur[i0 - 1]) / (2.0 * dx);
    double vxx = (Vcur[i0 + 1] - 2.0 * Vcur[i0] + Vcur[i0 - 1]) / (dx * dx);

    result_t res;
    res.price   = Vcur[i0];
    res.delta   = vx / S[i0];
    res.gamma   = (vxx - vx) / (S[i0] * S[i0]);
    res.seconds = t1 - t0;
    res.lattice_updates = (long long)(N - 1) * (long long)M;

    free(S); free(Vcur); free(Vnext); free(intr);
    return res;
}

/* Thomas algorithm: O(n) tridiagonal solve. Inherently sequential (the
 * forward sweep carries a loop-carried dependency), which is exactly the
 * argument for choosing the explicit scheme as the parallel target and
 * treating Crank-Nicolson as the accuracy reference. Milestone 3 can revisit
 * this with parallel cyclic reduction (PCR) if time allows. */
static void thomas(int n, const double *lo, const double *dg,
                   const double *up, const double *rhs, double *x,
                   double *cp, double *dp)
{
    cp[0] = up[0] / dg[0];
    dp[0] = rhs[0] / dg[0];
    for (int i = 1; i < n; ++i) {
        double m = dg[i] - lo[i] * cp[i - 1];
        cp[i] = up[i] / m;
        dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / m;
    }
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i)
        x[i] = dp[i] - cp[i] * x[i + 1];
}

/* Crank–Nicolson: second order in time, unconditionally stable, so M can be
 * chosen for accuracy rather than stability. */
static result_t solve_cn_1d(const params_t *p_in)
{
    params_t p = *p_in;
    if (p.N % 2) p.N += 1;

    int    N    = p.N;
    double half = p.L * p.sigma * sqrt(p.T);
    double xmin = log(p.S0) - half, xmax = log(p.S0) + half;
    double dx   = (xmax - xmin) / N;
    int    M    = p.M > 0 ? p.M : N;             /* dt ~ dx is fine for CN */
    double dt   = p.T / M;
    int    n    = N - 1;                         /* interior unknowns */

    double *S    = xmalloc((size_t)(N + 1) * sizeof(double));
    double *V    = xmalloc((size_t)(N + 1) * sizeof(double));
    double *intr = xmalloc((size_t)(N + 1) * sizeof(double));
    double *lo   = xmalloc((size_t)n * sizeof(double));
    double *dg   = xmalloc((size_t)n * sizeof(double));
    double *up   = xmalloc((size_t)n * sizeof(double));
    double *rhs  = xmalloc((size_t)n * sizeof(double));
    double *sol  = xmalloc((size_t)n * sizeof(double));
    double *cp   = xmalloc((size_t)n * sizeof(double));
    double *dp   = xmalloc((size_t)n * sizeof(double));

    for (int i = 0; i <= N; ++i) {
        S[i]    = exp(xmin + i * dx);
        intr[i] = payoff(S[i], p.K, p.is_call);
        V[i]    = intr[i];
    }

    double A = p.sigma * p.sigma / (2.0 * dx * dx);
    double B = (p.r - p.q - 0.5 * p.sigma * p.sigma) / (2.0 * dx);
    double Llo = A - B, Ldg = -2.0 * A - p.r, Lup = A + B;   /* operator L */
    double th = 0.5;

    for (int i = 0; i < n; ++i) {
        lo[i] = -th * dt * Llo;
        dg[i] = 1.0 - th * dt * Ldg;
        up[i] = -th * dt * Lup;
    }

    double t0 = now_sec();
    for (int k = 0; k < M; ++k) {
        double tau_new = (k + 1) * dt;
        double dfr_new = exp(-p.r * tau_new), dfq_new = exp(-p.q * tau_new);

        /* boundary values at the new time level */
        double b0, bN;
        if (p.is_call) {
            b0 = 0.0;
            bN = S[N] * dfq_new - p.K * dfr_new; if (bN < 0.0) bN = 0.0;
        } else {
            b0 = p.K * dfr_new - S[0] * dfq_new; if (b0 < 0.0) b0 = 0.0;
            bN = 0.0;
        }

        for (int i = 1; i <= n; ++i) {
            double e = (1.0 - th) * dt;
            rhs[i - 1] = V[i] + e * (Llo * V[i - 1] + Ldg * V[i] + Lup * V[i + 1]);
        }
        rhs[0]     -= lo[0]     * b0;   /* fold known boundaries into RHS */
        rhs[n - 1] -= up[n - 1] * bN;

        thomas(n, lo, dg, up, rhs, sol, cp, dp);

        V[0] = b0; V[N] = bN;
        for (int i = 1; i <= n; ++i) V[i] = sol[i - 1];
        if (p.is_american)
            for (int i = 0; i <= N; ++i) if (V[i] < intr[i]) V[i] = intr[i];
    }
    double t1 = now_sec();

    int i0 = N / 2;
    double vx  = (V[i0 + 1] - V[i0 - 1]) / (2.0 * dx);
    double vxx = (V[i0 + 1] - 2.0 * V[i0] + V[i0 - 1]) / (dx * dx);

    result_t res;
    res.price   = V[i0];
    res.delta   = vx / S[i0];
    res.gamma   = (vxx - vx) / (S[i0] * S[i0]);
    res.seconds = t1 - t0;
    res.lattice_updates = (long long)n * (long long)M;

    free(S); free(V); free(intr); free(lo); free(dg); free(up);
    free(rhs); free(sol); free(cp); free(dp);
    return res;
}

/* --------------------------------------------------------------------------
 * 4. 2D solver — two correlated assets (9-point stencil)
 *
 *   x = ln S1, y = ln S2
 *   V_tau = 1/2 s1^2 V_xx + 1/2 s2^2 V_yy + rho s1 s2 V_xy
 *           + (r-q1-s1^2/2) V_x + (r-q2-s2^2/2) V_y - r V
 *
 * The correlation term V_xy is what turns a 5-point stencil into a 9-point
 * one, and it is the reason the 3-asset (3D) case in Milestone 3 becomes a
 * 27-point stencil with heavy halo traffic.
 *
 * Validated against Margrabe's exchange option, payoff max(S1 - S2, 0).
 * --------------------------------------------------------------------------*/

typedef struct {
    double S1, S2, q1, q2, s1, s2, rho, r, T;
    int    N;      /* nodes per dimension - 1 */
    double L;
} params2d_t;

/* PARALLEL HOTSPOT #3 — the 9-point sweep. This is the kernel that gets
 * shared-memory tiling in CUDA (Milestone 3) and MPI halo exchange when the
 * grid is decomposed across nodes. */
static void step_explicit_2d(const double *restrict Vin, double *restrict Vout,
                             int N, const double w[9])
{
    int S = N + 1;
    for (int j = 1; j < N; ++j) {
        const double *rm = Vin + (size_t)(j - 1) * S;
        const double *r0 = Vin + (size_t)j       * S;
        const double *rp = Vin + (size_t)(j + 1) * S;
        double *o = Vout + (size_t)j * S;
        for (int i = 1; i < N; ++i) {
            o[i] = w[0] * r0[i]
                 + w[1] * r0[i - 1] + w[2] * r0[i + 1]
                 + w[3] * rm[i]     + w[4] * rp[i]
                 + w[5] * rm[i - 1] + w[6] * rp[i + 1]
                 + w[7] * rm[i + 1] + w[8] * rp[i - 1];
        }
    }
}

static double solve_exchange_2d(const params2d_t *p_in, double *secs,
                                long long *updates)
{
    params2d_t p = *p_in;
    if (p.N % 2) p.N += 1;
    int N = p.N, S = N + 1;

    double hx = p.L * p.s1 * sqrt(p.T), hy = p.L * p.s2 * sqrt(p.T);
    double xmin = log(p.S1) - hx, ymin = log(p.S2) - hy;
    double dx = 2.0 * hx / N, dy = 2.0 * hy / N;

    /* stability bound including the cross term */
    double bound = p.s1 * p.s1 / (dx * dx) + p.s2 * p.s2 / (dy * dy)
                 + fabs(p.rho) * p.s1 * p.s2 / (dx * dy) + p.r;
    int    M  = (int)ceil(p.T * bound / 0.9);
    if (M < 8) M = 8;
    double dt = p.T / M;

    double *S1 = xmalloc((size_t)S * sizeof(double));
    double *S2 = xmalloc((size_t)S * sizeof(double));
    double *Vc = xmalloc((size_t)S * S * sizeof(double));
    double *Vn = xmalloc((size_t)S * S * sizeof(double));

    for (int i = 0; i < S; ++i) { S1[i] = exp(xmin + i * dx); S2[i] = exp(ymin + i * dy); }
    for (int j = 0; j < S; ++j)
        for (int i = 0; i < S; ++i) {
            double v = S1[i] - S2[j];
            Vc[(size_t)j * S + i] = v > 0.0 ? v : 0.0;
        }

    double dxx = 0.5 * p.s1 * p.s1 / (dx * dx);
    double dyy = 0.5 * p.s2 * p.s2 / (dy * dy);
    double dxy = p.rho * p.s1 * p.s2 / (4.0 * dx * dy);
    double cx  = (p.r - p.q1 - 0.5 * p.s1 * p.s1) / (2.0 * dx);
    double cy  = (p.r - p.q2 - 0.5 * p.s2 * p.s2) / (2.0 * dy);

    double w[9];
    w[0] = 1.0 + dt * (-2.0 * dxx - 2.0 * dyy - p.r);
    w[1] = dt * (dxx - cx);          /* (i-1, j) */
    w[2] = dt * (dxx + cx);          /* (i+1, j) */
    w[3] = dt * (dyy - cy);          /* (i, j-1) */
    w[4] = dt * (dyy + cy);          /* (i, j+1) */
    w[5] = dt *  dxy;                /* (i-1, j-1) */
    w[6] = dt *  dxy;                /* (i+1, j+1) */
    w[7] = dt * -dxy;                /* (i+1, j-1) */
    w[8] = dt * -dxy;                /* (i-1, j+1) */

    double t0 = now_sec();
    for (int n = 0; n < M; ++n) {
        double tau = (n + 1) * dt;
        double f1 = exp(-p.q1 * tau), f2 = exp(-p.q2 * tau);
        step_explicit_2d(Vc, Vn, N, w);

        /* Dirichlet boundaries (extreme market conditions):
         *   S1 -> 0   : never exercise            -> 0
         *   S1 -> inf : deep in the money         -> S1 e^-q1 tau - S2 e^-q2 tau
         *   S2 -> 0   : receive asset 1 for free  -> S1 e^-q1 tau
         *   S2 -> inf : never exercise            -> 0                       */
        for (int j = 0; j < S; ++j) {
            Vn[(size_t)j * S + 0] = 0.0;
            double v = S1[N] * f1 - S2[j] * f2;
            Vn[(size_t)j * S + N] = v > 0.0 ? v : 0.0;
        }
        for (int i = 0; i < S; ++i) {
            Vn[(size_t)0 * S + i] = S1[i] * f1;
            Vn[(size_t)N * S + i] = 0.0;
        }
        double *t = Vc; Vc = Vn; Vn = t;
    }
    double t1 = now_sec();

    double price = Vc[(size_t)(N / 2) * S + (N / 2)];
    *secs    = t1 - t0;
    *updates = (long long)(N - 1) * (N - 1) * M;

    free(S1); free(S2); free(Vc); free(Vn);
    return price;
}

/* --------------------------------------------------------------------------
 * 5. Drivers
 * --------------------------------------------------------------------------*/

static void mode_price(void)
{
    params_t p = default_params();
    greeks_t an = bs_analytic(p.S0, p.K, p.r, p.q, p.sigma, p.T, p.is_call);

    result_t ex = solve_explicit_1d(&p);
    result_t cn = solve_cn_1d(&p);

    printf("European call  S0=%.1f K=%.1f r=%.3f sigma=%.2f T=%.1f  N=%d\n\n",
           p.S0, p.K, p.r, p.sigma, p.T, p.N);
    printf("%-16s %14s %14s %14s\n", "quantity", "analytical", "explicit", "Crank-Nic");
    printf("%-16s %14.8f %14.8f %14.8f\n", "price", an.price, ex.price, cn.price);
    printf("%-16s %14s %14.2e %14.2e\n", "abs error", "-",
           fabs(ex.price - an.price), fabs(cn.price - an.price));
    printf("%-16s %14.8f %14.8f %14.8f\n", "delta", an.delta, ex.delta, cn.delta);
    printf("%-16s %14.8f %14.8f %14.8f\n", "gamma", an.gamma, ex.gamma, cn.gamma);
    printf("\n%-16s %14s %14.4f %14.4f  (seconds)\n", "wall time", "-",
           ex.seconds, cn.seconds);
    printf("%-16s %14s %14lld %14lld\n", "lattice updates", "-",
           ex.lattice_updates, cn.lattice_updates);
}

static void mode_american(void)
{
    params_t p = default_params();
    p.is_call = 0;                 /* American put: early exercise matters */
    p.N = 1024;

    greeks_t an = bs_analytic(p.S0, p.K, p.r, p.q, p.sigma, p.T, 0);
    p.is_american = 0; result_t eu = solve_explicit_1d(&p);
    p.is_american = 1; result_t am = solve_explicit_1d(&p);

    printf("Put option  S0=%.1f K=%.1f r=%.3f sigma=%.2f T=%.1f  N=%d\n\n",
           p.S0, p.K, p.r, p.sigma, p.T, p.N);
    printf("European (analytical) : %.8f\n", an.price);
    printf("European (FDM)        : %.8f   err %.2e\n", eu.price,
           fabs(eu.price - an.price));
    printf("American (FDM)        : %.8f\n", am.price);
    printf("Early-exercise premium: %.8f\n", am.price - eu.price);
    printf("\n(premium must be > 0 for an American put; this is the "
           "qualitative check\n when no closed form exists)\n");
}

static void mode_converge(void)
{
    params_t p = default_params();
    greeks_t an = bs_analytic(p.S0, p.K, p.r, p.q, p.sigma, p.T, p.is_call);

    printf("Spatial convergence, explicit Euler (dt set by stability => "
           "dt ~ dx^2)\n");
    printf("Reference price = %.10f\n\n", an.price);
    printf("%8s %10s %14s %14s %10s %12s\n",
           "N", "M", "dx", "abs error", "order", "time (s)");

    double prev_err = 0.0, prev_dx = 0.0;
    for (int N = 128; N <= 2048; N *= 2) {
        p.N = N;
        double half = p.L * p.sigma * sqrt(p.T);
        double dx = 2.0 * half / N;
        int    M = auto_timesteps(&p, dx);
        result_t r = solve_explicit_1d(&p);
        double err = fabs(r.price - an.price);

        if (prev_err > 0.0) {
            double order = log(prev_err / err) / log(prev_dx / dx);
            printf("%8d %10d %14.3e %14.3e %10.2f %12.4f\n",
                   N, M, dx, err, order, r.seconds);
        } else {
            printf("%8d %10d %14.3e %14.3e %10s %12.4f\n",
                   N, M, dx, err, "-", r.seconds);
        }
        prev_err = err; prev_dx = dx;
    }
    printf("\nOrder should approach 2.0 (second-order central differences in "
           "space).\n");
}

static void mode_bench(void)
{
    params_t p = default_params();
    greeks_t an = bs_analytic(p.S0, p.K, p.r, p.q, p.sigma, p.T, p.is_call);

    /* 5 flops per interior node per step: 3 multiplies + 2 adds.
     * 16 bytes of compulsory traffic per node per step (8 read + 8 write;
     * the two neighbours are served from cache in a streaming sweep).
     * => arithmetic intensity ~ 0.31 flop/byte: firmly memory bound. */
    const double FLOP_PER_UPDATE  = 5.0;
    const double BYTES_PER_UPDATE = 16.0;

    printf("Sequential baseline, explicit Euler 3-point stencil (1 thread)\n");
    printf("flops/update = %.0f   bytes/update = %.0f   "
           "arithmetic intensity = %.3f flop/byte\n\n",
           FLOP_PER_UPDATE, BYTES_PER_UPDATE, FLOP_PER_UPDATE / BYTES_PER_UPDATE);
    printf("%8s %10s %16s %12s %10s %12s %12s\n",
           "N", "M", "updates", "time (s)", "MLUP/s", "GFLOP/s", "GB/s");

    int sizes[] = {256, 512, 1024, 2048};
    for (unsigned k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
        p.N = sizes[k];
        result_t r = solve_explicit_1d(&p);
        double lu = (double)r.lattice_updates;
        double half = p.L * p.sigma * sqrt(p.T);
        int    M = auto_timesteps(&p, 2.0 * half / p.N);
        printf("%8d %10d %16lld %12.4f %10.1f %12.3f %12.2f\n",
               p.N, M, r.lattice_updates, r.seconds,
               lu / r.seconds / 1e6,
               lu * FLOP_PER_UPDATE / r.seconds / 1e9,
               lu * BYTES_PER_UPDATE / r.seconds / 1e9);
        (void)an;
    }
    printf("\nWork scales as O(N^3): doubling N doubles the nodes and "
           "quadruples the\ntime steps required for stability.\n");
}

static void mode_2d(void)
{
    params2d_t p;
    p.S1 = 100.0; p.S2 = 95.0; p.q1 = 0.0; p.q2 = 0.0;
    p.s1 = 0.25;  p.s2 = 0.30; p.rho = 0.40;
    p.r  = 0.05;  p.T  = 1.0;  p.L = 4.0;

    printf("Two correlated assets, exchange option max(S1 - S2, 0)\n");
    printf("S1=%.1f S2=%.1f s1=%.2f s2=%.2f rho=%.2f T=%.1f\n\n",
           p.S1, p.S2, p.s1, p.s2, p.rho, p.T);

    double ref = margrabe(p.S1, p.S2, p.q1, p.q2, p.s1, p.s2, p.rho, p.T);
    printf("Margrabe closed form : %.8f\n\n", ref);
    printf("%8s %10s %16s %14s %12s %12s %10s\n",
           "N", "M", "updates", "FDM price", "abs error", "time (s)", "MLUP/s");

    for (int N = 64; N <= 512; N *= 2) {
        p.N = N;
        double secs; long long upd;
        double v = solve_exchange_2d(&p, &secs, &upd);
        printf("%8d %10lld %16lld %14.8f %12.2e %12.4f %10.1f\n",
               N, upd / ((long long)(N - 1) * (N - 1)), upd, v,
               fabs(v - ref), secs, (double)upd / secs / 1e6);
    }
    printf("\n9-point stencil (the 4 diagonal weights come from the "
           "correlation term).\nThe 3-asset case in Milestone 3 becomes a "
           "27-point stencil in 3D.\n");
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "price";
    if      (!strcmp(mode, "price"))    mode_price();
    else if (!strcmp(mode, "american")) mode_american();
    else if (!strcmp(mode, "converge")) mode_converge();
    else if (!strcmp(mode, "bench"))    mode_bench();
    else if (!strcmp(mode, "2d"))       mode_2d();
    else {
        fprintf(stderr, "usage: %s [price|american|converge|bench|2d]\n", argv[0]);
        return 1;
    }
    return 0;
}
