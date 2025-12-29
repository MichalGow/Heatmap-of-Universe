import numpy as np
import matplotlib.pyplot as plt
import itertools

# ---------------------------
# Angle/grid helpers
# ---------------------------
def make_angle_grid(n_cols: int, high_deg: float = 180.0) -> np.ndarray:
    if n_cols < 2:
        raise ValueError("n_cols must be >= 2")
    return np.linspace(0.0, float(high_deg), int(n_cols))

def deg_to_idx(deg: float, n_cols: int, high_deg: float = 180.0) -> int:
    if not (0.0 <= deg <= high_deg):
        raise ValueError(f"deg must be in [0, {high_deg}]")
    return int(round((deg / high_deg) * (n_cols - 1)))

def idx_to_deg(idx: int, n_cols: int, high_deg: float = 180.0) -> float:
    if not (0 <= idx < n_cols):
        raise ValueError(f"idx must be in [0, {n_cols-1}]")
    return (idx / (n_cols - 1)) * high_deg

def compute_selected_cols(degrees, n_cols: int, high_deg: float = 180.0):
    degrees = list(map(float, degrees))
    idxs = [deg_to_idx(d, n_cols, high_deg) for d in degrees]
    approx = [idx_to_deg(i, n_cols, high_deg) for i in idxs]
    return tuple(idxs), np.array(approx, dtype=float)

# ---------------------------
# Bell-ish match target: cos^2(delta)
# ---------------------------
def mirror_delta_deg(a_deg: float, b_deg: float, period_deg: float = 180.0) -> float:
    d = abs(a_deg - b_deg) % period_deg
    return min(d, period_deg - d)

def p_match_cos2(delta_deg: float) -> float:
    return float(np.cos(np.deg2rad(delta_deg)) ** 2)

def sample_pair_outcomes(rng: np.random.Generator, p_match: float):
    """
    Sample (A,B) in {0,1}^2 with:
      P(A=1)=P(B=1)=0.5 and P(A==B)=p_match
    """
    u = rng.random()
    if u < 0.5 * p_match:
        return 0, 0
    elif u < p_match:
        return 1, 1
    elif u < p_match + 0.5 * (1.0 - p_match):
        return 0, 1
    else:
        return 1, 0

# ---------------------------
# Full table + mask generator
# ---------------------------
def make_full_matrix_with_mask(
    n_rows: int,
    n_cols: int,
    selected_cols: tuple[int, ...],
    col_settings_deg: np.ndarray,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)

    selected_cols = list(map(int, selected_cols))
    pairs = list(itertools.combinations(selected_cols, 2))

    # Full unbiased matrix -> whole row/col ~50/50
    X = rng.integers(0, 2, size=(n_rows, n_cols), dtype=np.int8)

    # Mask indicates which cells were actually "measured" in that row
    M = np.zeros((n_rows, n_cols), dtype=np.int8)

    # Choose which pair is measured per row, uniformly across pairs
    pair_idx = rng.integers(0, len(pairs), size=n_rows)
    for r in range(n_rows):
        i, j = pairs[pair_idx[r]]

        ai = float(col_settings_deg[i])
        aj = float(col_settings_deg[j])
        delta = mirror_delta_deg(ai, aj, period_deg=180.0)
        pm = p_match_cos2(delta)

        A, B = sample_pair_outcomes(rng, pm)
        X[r, i] = A
        X[r, j] = B
        M[r, i] = 1
        M[r, j] = 1

    return X, M, pairs

def pairwise_match_rates_on_measured_rows(X: np.ndarray, M: np.ndarray, pairs):
    out = []
    for i, j in pairs:
        valid = (M[:, i] == 1) & (M[:, j] == 1)
        match = float((X[valid, i] == X[valid, j]).mean())
        out.append((i, j, match, int(valid.sum())))
    return out

# ---------------------------
# Demo runner + plots
# ---------------------------
def run_case(n_cols: int, degrees=(0, 45, 90, 180), n_rows: int = 20_000, show_rows: int = 2000, seed: int = 7):
    high_deg = 180.0
    col_settings = make_angle_grid(n_cols, high_deg=high_deg)
    selected_cols, approx_degs = compute_selected_cols(degrees, n_cols, high_deg=high_deg)

    X, M, pairs = make_full_matrix_with_mask(
        n_rows=n_rows,
        n_cols=n_cols,
        selected_cols=selected_cols,
        col_settings_deg=col_settings,
        seed=seed,
    )

    # Print mapping + match rates
    print(f"\n=== n_cols={n_cols} ===")
    print("Requested degrees:", list(degrees))
    print("Selected col indices:", list(selected_cols))
    print("Actual degrees at those indices:", [round(float(d), 6) for d in approx_degs])
    print("Angle step (deg/col):", high_deg / (n_cols - 1))

    # Whole-matrix marginals
    overall = float(X.mean())
    row_mean = X.mean(axis=1)
    col_mean = X.mean(axis=0)
    print(f"Overall P(1) in full X: {overall:.4f}")
    print(f"Row P(1) range: {row_mean.min():.3f} .. {row_mean.max():.3f} (avg {row_mean.mean():.4f})")
    print(f"Col P(1) range: {col_mean.min():.3f} .. {col_mean.max():.3f} (avg {col_mean.mean():.4f})")

    # Pairwise match rates vs cos^2(target delta)
    results = pairwise_match_rates_on_measured_rows(X, M, pairs)
    print("\nPairwise match rates (computed only on rows where that pair was measured):")
    for i, j, match, n_used in results:
        delta = mirror_delta_deg(col_settings[i], col_settings[j], period_deg=180.0)
        target = p_match_cos2(delta)
        print(f"  ({i}-{j})  Δ≈{delta:7.3f}°  match={match:.4f}  target={target:.4f}  rows_used={n_used}")

    # Heatmaps (no grouping)
    show_rows = min(show_rows, n_rows)

    # Mask heatmap shows the "measurement pattern"
    plt.figure(figsize=(12, 6))
    plt.imshow(M[:show_rows, :], aspect="auto", interpolation="nearest", origin="lower")
    plt.colorbar(label="Measured? (0/1)")
    plt.title(f"Mask heatmap (no grouping): first {show_rows} rows × {n_cols} cols")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.tight_layout()
    plt.show()

    # Outcomes heatmap for ALL columns (no grouping)
    plt.figure(figsize=(14, 6))
    plt.imshow(X[:show_rows, :], aspect="auto", interpolation="nearest", origin="lower")
    plt.colorbar(label="Outcome (0/1)")
    plt.title(f"Outcomes heatmap (ALL cols): first {show_rows} rows × {n_cols} cols")
    plt.xlabel("Column index")
    plt.ylabel("Row index")

    # Keep x-ticks readable
    tick_step = max(1, n_cols // 12)
    plt.xticks(np.arange(0, n_cols, tick_step), np.arange(0, n_cols, tick_step))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for cols in (20, 360, 720):
        run_case(n_cols=cols, degrees=(0, 22.5, 45, 67.5), n_rows=20_000, show_rows=2000, seed=7)
