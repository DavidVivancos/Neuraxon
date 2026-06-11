# Neuraxon Game of Life v.5.05 gfactor (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 190
"""
g-factor measurement for the Game of Life — v179 (v4.87).

Implements the methodology of an upcoming paper on cross-battery
dissociation of the general factor (g) in a lesionable artificial
cognitive system, adapted to Neuraxon's continuous survival simulation.

The paper computes g as the dominant factor (PC1) of the population
correlation matrix over a battery of CHC-aligned cognitive-ability scores,
plus the positive-manifold fraction, KMO, and the λ1/λ2 unidimensionality
indicator. The Game of Life has no discrete psychometric battery, so we
derive a battery of per-NxEr *ability proxies* from quantities the
simulation already tracks (no change to dynamics — this module is
read-only over the living population), then extract the same statistics.

CHC-aligned ability-proxy battery (one scalar per living NxEr):

    proxy        CHC factor   derived from                         paper analogue
    ----------   ----------   ----------------------------------   --------------
    Gs_speed     Gs           food_taken per second lived          processing speed
    Gv_spatial   Gv           unique cells visited per second      visual/spatial
    Gf_fluid     Gf           novel-food discovery rate            fluid reasoning
    Glr_memory   Glr          distinct food sources remembered     long-term retrieval
    Gc_social    Gc           lifetime mates performed             crystallised/social
    Gsm_wm       Gsm          branching-ratio proximity to 1       short-term/working memory
    Ga_audio     Ga           song/sing channel engagement         auditory processing

This is a deliberately faithful analogue: the proxies span distinguishable
competencies that *can* be uncorrelated, so a positive manifold (if it
emerges) is an emergent property of the architecture's shared dynamics —
exactly the question the user wants to probe ("can the Game-of-Life setup
reproduce g?"). Per the paper, g is a *population* construct: per-NxEr g
is the agent's standardised-proxy projection onto PC1, not an intrinsic
scalar.

All outputs are NaN-safe and degrade gracefully below the minimum sample
size (returns zeros), so wiring this into the metrics pipeline can never
crash a run.
"""
from __future__ import annotations

import math
from typing import List, Dict, Any

try:
    import numpy as _np
    _HAVE_NUMPY = True
except Exception:                                    # pragma: no cover
    _HAVE_NUMPY = False

# Minimum living-population size for a meaningful factor analysis. Below
# this the correlation matrix is too noisy; the paper itself flags N≈500
# as the lower bound for stable bifactor CFA — we only need a stable PC1
# so a much smaller floor is acceptable, but below ~8 it is meaningless.
MIN_AGENTS_FOR_G = 8

# Stable proxy order. CHANGING THIS ORDER CHANGES NOTHING numerically (the
# correlation matrix is order-invariant) but keeps per-proxy logging stable.
PROXY_NAMES = [
    "Gs_speed", "Gv_spatial", "Gf_fluid", "Glr_memory",
    "Gc_social", "Gsm_wm", "Ga_audio",
]


def _safe(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _agent_age_seconds(nxer) -> float:
    """Seconds this NxEr has been alive (>= 1.0 to avoid div-by-zero)."""
    st = getattr(nxer, "stats", None)
    if st is not None:
        tl = _safe(getattr(st, "time_lived_s", 0.0))
        if tl > 1.0:
            return tl
    # Fallback to wall-clock lifespan.
    try:
        import time as _t
        born = getattr(nxer, "born_ts", None)
        if born is not None:
            return max(1.0, _t.time() - float(born))
    except Exception:
        pass
    return 1.0


def extract_ability_vector(nxer) -> List[float]:
    """Return the 7-proxy CHC-aligned ability vector for one NxEr.

    Pure read of already-tracked state — no side effects, no dynamics.
    """
    st = getattr(nxer, "stats", None)
    age_s = _agent_age_seconds(nxer)

    food_taken = _safe(getattr(st, "food_taken", 0.0)) if st else 0.0
    explored = _safe(getattr(st, "explored", 0.0)) if st else 0.0
    mates = _safe(getattr(st, "mates_performed", 0.0)) if st else 0.0

    visited = getattr(nxer, "visited", None)
    n_visited = float(len(visited)) if visited else explored
    known_food = getattr(nxer, "known_food_ids", None)
    n_known_food = float(len(known_food)) if known_food else 0.0

    # Gsm — working-memory proxy: how close the recurrent network sits to
    # criticality (branching ratio ≈ 1.0 ⇒ optimal information retention).
    net = getattr(nxer, "net", None)
    br = _safe(getattr(net, "branching_ratio", 0.0)) if net is not None else 0.0
    gsm = 1.0 / (1.0 + abs(br - 1.0)) if br > 0.0 else 0.0

    # Ga — auditory engagement: absolute sing level integrated as a simple
    # proxy for use of the song/hearing channel.
    sing = abs(_safe(getattr(nxer, "last_sing_level", 0.0)))

    return [
        food_taken / age_s,                 # Gs_speed
        n_visited / age_s,                  # Gv_spatial
        n_known_food / age_s,               # Gf_fluid (discovery rate)
        n_known_food,                       # Glr_memory (accumulated)
        mates,                              # Gc_social
        gsm,                                # Gsm_wm
        sing,                               # Ga_audio
    ]


def _zero_result() -> Dict[str, Any]:
    return {
        "g_pc1_fraction": 0.0,
        "g_positive_manifold": 0.0,
        "g_mean_offdiag_r": 0.0,
        "g_lambda1_over_lambda2": 0.0,
        "g_n_agents": 0,
    }


def compute_population_g(alive_nxers: List[Any],
                         write_back: bool = True) -> Dict[str, Any]:
    """Compute the population g signatures over the living NxErs.

    Mirrors the paper's analysis pipeline:
      * build the N×7 ability matrix,
      * z-score each proxy column,
      * correlation matrix R,
      * PC1 eigenvalue fraction (paper's "PC1%"),
      * positive-manifold fraction (paper's "% positive"),
      * mean off-diagonal r (paper's "mean r"),
      * λ1/λ2 (paper's unidimensionality indicator; ≈1 ⇒ no unitary g).

    When ``write_back`` is True, each NxEr gets a transient ``_g_score``
    attribute = its standardised ability vector projected onto PC1 (the
    per-agent factor score; this is the paper's per-agent g definition).

    Returns a dict of scalars (all finite). Never raises.
    """
    try:
        agents = [a for a in alive_nxers if getattr(a, "alive", True)]
        n = len(agents)
        if n < MIN_AGENTS_FOR_G:
            return _zero_result()

        rows = [extract_ability_vector(a) for a in agents]
        k = len(PROXY_NAMES)

        if _HAVE_NUMPY:
            X = _np.asarray(rows, dtype=float)            # N × K
            X = _np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            live = sd > 1e-9                              # proxies with variance
            if int(live.sum()) < 2:
                return _zero_result()
            Xs = (X[:, live] - mu[live]) / sd[live]
            R = _np.corrcoef(Xs, rowvar=False)
            R = _np.nan_to_num(R, nan=0.0)
            kk = R.shape[0]
            eig = _np.linalg.eigvalsh(R)                  # ascending
            eig = _np.sort(eig)[::-1]
            lam1 = float(eig[0])
            lam2 = float(eig[1]) if kk > 1 else 0.0
            trace = float(_np.trace(R)) or float(kk)
            pc1_frac = max(0.0, lam1 / trace)
            iu = _np.triu_indices(kk, k=1)
            off = R[iu]
            pos_manifold = float((off > 0).mean()) if off.size else 0.0
            mean_off = float(off.mean()) if off.size else 0.0
            l1l2 = (lam1 / lam2) if lam2 > 1e-9 else float(kk)

            if write_back:
                # PC1 eigenvector via eigh on R (symmetric).
                w, V = _np.linalg.eigh(R)
                pc1 = V[:, int(_np.argmax(w))]
                scores = Xs.dot(pc1)
                # Orient so higher score = "more able" (positive mean loading).
                if pc1.sum() < 0:
                    scores = -scores
                for a, s in zip(agents, scores):
                    try:
                        a._g_score = float(s)
                        # v180 — mirror onto stats so the existing rankings /
                        # all_time_best / round-carryover machinery (which all
                        # read getattr(a.stats, <name>)) handles g uniformly.
                        if getattr(a, 'stats', None) is not None:
                            a.stats.g_factor = float(s)
                    except Exception:
                        pass
            return {
                "g_pc1_fraction": _safe(pc1_frac),
                "g_positive_manifold": _safe(pos_manifold),
                "g_mean_offdiag_r": _safe(mean_off),
                "g_lambda1_over_lambda2": _safe(min(l1l2, 99.0)),
                "g_n_agents": n,
            }

        # ---- pure-python fallback (no numpy) ----
        cols = list(zip(*rows))
        means = [sum(c) / n for c in cols]
        sds = [math.sqrt(sum((v - means[j]) ** 2 for v in c) / n)
               for j, c in enumerate(cols)]
        idx = [j for j in range(k) if sds[j] > 1e-9]
        if len(idx) < 2:
            return _zero_result()
        Z = [[(rows[i][j] - means[j]) / sds[j] for j in idx]
             for i in range(n)]
        m = len(idx)
        R = [[0.0] * m for _ in range(m)]
        for a in range(m):
            for b in range(m):
                R[a][b] = sum(Z[i][a] * Z[i][b] for i in range(n)) / n
        offs = [R[a][b] for a in range(m) for b in range(a + 1, m)]
        pos_manifold = (sum(1 for v in offs if v > 0) / len(offs)
                        if offs else 0.0)
        mean_off = sum(offs) / len(offs) if offs else 0.0
        # Power iteration for the dominant eigenvalue/vector of R.
        v = [1.0 / math.sqrt(m)] * m
        lam1 = 0.0
        for _ in range(80):
            w = [sum(R[a][b] * v[b] for b in range(m)) for a in range(m)]
            norm = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / norm for x in w]
            lam1 = norm
        trace = sum(R[a][a] for a in range(m)) or float(m)
        pc1_frac = max(0.0, lam1 / trace)
        # crude λ2 via deflation
        Rd = [[R[a][b] - lam1 * v[a] * v[b] for b in range(m)]
              for a in range(m)]
        u = [1.0 / math.sqrt(m)] * m
        lam2 = 0.0
        for _ in range(80):
            w = [sum(Rd[a][b] * u[b] for b in range(m)) for a in range(m)]
            norm = math.sqrt(sum(x * x for x in w)) or 1.0
            u = [x / norm for x in w]
            lam2 = norm
        l1l2 = (lam1 / lam2) if lam2 > 1e-9 else float(m)
        if write_back:
            for i, ag in enumerate(agents):
                try:
                    _gs = float(sum(Z[i][j] * v[j] for j in range(m)))
                    ag._g_score = _gs
                    if getattr(ag, 'stats', None) is not None:
                        ag.stats.g_factor = _gs   # v180 — uniform ranking/carryover
                except Exception:
                    pass
        return {
            "g_pc1_fraction": _safe(pc1_frac),
            "g_positive_manifold": _safe(pos_manifold),
            "g_mean_offdiag_r": _safe(mean_off),
            "g_lambda1_over_lambda2": _safe(min(l1l2, 99.0)),
            "g_n_agents": n,
        }
    except Exception:
        # g measurement must NEVER break a run.
        return _zero_result()
