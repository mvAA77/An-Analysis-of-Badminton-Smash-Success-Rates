import os
from pathlib import Path
import argparse
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score

import matplotlib.pyplot as plt
import matplotlib.patches as patches


SMASH_KEYWORDS = ["殺球", "點扣", "smash"]
DEFAULT_SET_FOLDER = "ShuttleSet/set"
OUTPUT_DIR = "smash_analysis"

COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4

DEFAULT_TARGET_PRECISION = 0.85
USE_GROUP_SPLIT = True
DROP_OUT_OF_COURT = False
RANDOM_STATE = 42

def infer_match_mode_from_name(match_name: str) -> str:
    name_lower = str(match_name).lower()
    doubles_keywords = ["md", "wd", "xd", "double", "doubles", "mixed"]
    return "doubles" if any(k in name_lower for k in doubles_keywords) else "singles"

def is_in_rally_court(landing_x_m: float, landing_y_m: float, mode: str) -> bool:
    if mode == "doubles":
        return (0.0 <= landing_x_m <= COURT_WIDTH_M) and (0.0 <= landing_y_m <= COURT_LENGTH_M)
    left, right = 0.46, COURT_WIDTH_M - 0.46
    return (left <= landing_x_m <= right) and (0.0 <= landing_y_m <= COURT_LENGTH_M)

def draw_badminton_court(ax, mode="singles", width=6.1, length=13.4):
    outer = patches.Rectangle((0, 0), width, length, linewidth=2, edgecolor="black", facecolor="none")
    ax.add_patch(outer)
    net_y = length / 2
    ax.axhline(net_y, color="black", linestyle="--", linewidth=2)
    ax.hlines([net_y - 1.98, net_y + 1.98], 0, width, colors="gray", linestyles="-", linewidth=1.2)
    ax.vlines(width / 2, net_y - 1.98, net_y + 1.98, colors="gray", linestyles="-", linewidth=1.2)
    if mode == "singles":
        ax.vlines([0.46, width - 0.46], 0, length, colors="gray", linestyles="-", linewidth=1.2)
        ax.hlines([0.76, length - 0.76], 0.46, width - 0.46, colors="gray", linestyles="-", linewidth=1.0)
    else:
        ax.hlines([0.76, length - 0.76], 0, width, colors="gray", linestyles=":", linewidth=1.0)
    ax.text(width / 2, net_y + 0.25, "Opponent Side", ha="center", fontsize=9)
    ax.text(width / 2, net_y - 0.6, "Player Side", ha="center", fontsize=9)

def load_all_sets(base_path: Path) -> pd.DataFrame:
    all_match_dfs = []
    for match_dir in base_path.iterdir():
        if not match_dir.is_dir():
            continue
        for set_file in match_dir.glob("*.csv"):
            df = pd.read_csv(set_file, encoding="utf-8-sig")
            df.columns = [c.strip().lower() for c in df.columns]
            df["__match_name"] = match_dir.name
            df["__set_file"] = set_file.name
            all_match_dfs.append(df)
    return pd.concat(all_match_dfs, ignore_index=True)

def get_rally_winners(full_df: pd.DataFrame) -> pd.DataFrame:
    full_df = full_df.sort_values(["__match_name", "__set_file", "rally", "ball_round"], ascending=True)
    rally_winners = (
        full_df.groupby(["__match_name", "__set_file", "rally"])
        .tail(1)[["__match_name", "__set_file", "rally", "getpoint_player"]]
        .rename(columns={"getpoint_player": "rally_winner"})
    )
    return rally_winners

def get_all_smashes(full_df: pd.DataFrame, rally_winners: pd.DataFrame) -> pd.DataFrame:
    df = full_df.copy()
    df["ball_round"] = pd.to_numeric(df["ball_round"], errors="coerce")

    last_shots = (
        df.sort_values(["__match_name", "__set_file", "rally", "ball_round"], ascending=True)
          .groupby(["__match_name", "__set_file", "rally"], as_index=False)
          .tail(1)
          .copy()
    )

    def is_smash_type(x: str) -> bool:
        s = str(x).lower()
        return any(k.lower() in s for k in SMASH_KEYWORDS)

    last_smashes = last_shots[last_shots["type"].astype(str).apply(is_smash_type)].copy()

    last_smashes = last_smashes.merge(
        rally_winners,
        on=["__match_name", "__set_file", "rally"],
        how="left"
    )
    last_smashes["smash_success"] = (
        last_smashes["player"].astype(str) == last_smashes["rally_winner"].astype(str)
    ).astype(int)

    return last_smashes


def build_features_from_smashes(smashes: pd.DataFrame):
    smashes = smashes.dropna(subset=["landing_x", "landing_y", "player_location_x", "player_location_y"]).copy()
    px_x_min, px_x_max = smashes["landing_x"].min(), smashes["landing_x"].max()
    px_y_min, px_y_max = smashes["landing_y"].min(), smashes["landing_y"].max()
    x_scale = COURT_WIDTH_M / (px_x_max - px_x_min)
    y_scale = COURT_LENGTH_M / (px_y_max - px_y_min)
    for c in ["landing_x", "player_location_x"]:
        smashes[c + "_m"] = (smashes[c] - px_x_min) * x_scale
    for c in ["landing_y", "player_location_y"]:
        smashes[c + "_m"] = (smashes[c] - px_y_min) * y_scale
    smashes["distance_to_birdie_m"] = np.sqrt(
        (smashes["landing_x_m"] - smashes["player_location_x_m"]) ** 2 +
        (smashes["landing_y_m"] - smashes["player_location_y_m"]) ** 2
    )
    smashes["depth_m"] = smashes["landing_y_m"]
    smashes["width_m"] = smashes["landing_x_m"]
    smashes["to_center_m"] = np.abs(smashes["landing_x_m"] - (COURT_WIDTH_M / 2))
    smashes["to_back_m"] = COURT_LENGTH_M - smashes["landing_y_m"]
    smashes["depth_x_width"] = smashes["landing_y_m"] * smashes["to_center_m"]
    smashes["match_mode"] = smashes["__match_name"].apply(infer_match_mode_from_name)
    smashes["in_court"] = smashes.apply(lambda r: is_in_rally_court(r["landing_x_m"], r["landing_y_m"], r["match_mode"]), axis=1)
    smashes["in_court_flag"] = smashes["in_court"].astype(int)
    feature_cols = [
        "landing_x_m", "landing_y_m", "player_location_x_m", "player_location_y_m",
        "distance_to_birdie_m", "to_center_m", "to_back_m", "depth_m", "width_m",
        "depth_x_width", "in_court_flag"
    ]
    return smashes, smashes[feature_cols], smashes["smash_success"].astype(int), feature_cols

def split_train_test(smashes, X, y):
    if USE_GROUP_SPLIT:
        groups = smashes["__match_name"]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

def select_threshold(y_true, y_prob, mode="precision", target_precision=DEFAULT_TARGET_PRECISION):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if mode == "precision":
        p = prec[1:]
        r = rec[1:]
        t = thr

        valid = np.where(p >= target_precision)[0]
        if len(valid) == 0:
            f1s = 2 * (p * r) / (p + r + 1e-9)
            idx = int(np.argmax(f1s))
            return float(t[idx]), prec, rec, thr, idx + 1

        best_i = valid[np.argmax(r[valid])]
        return float(t[best_i]), prec, rec, thr, best_i + 1

    p = prec[1:]
    r = rec[1:]
    t = thr
    f1s = 2 * (p * r) / (p + r + 1e-9)
    idx = int(np.argmax(f1s))
    return float(t[idx]), prec, rec, thr, idx + 1


def train_smash_model_with_threshold(X, y, threshold_mode="precision", target_precision=DEFAULT_TARGET_PRECISION):
    X_train, X_test, y_train, y_test = split_train_test(smashes_global, X, y)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight={0: 1.0, 1: 2.0}, random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1.0, 3.0, 10.0]}, scoring="roc_auc", cv=5, n_jobs=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_prob = best.predict_proba(X_test)[:, 1]

    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1)

    best_threshold = thresholds[best_idx - 1]

    print("Using F1-optimized threshold:", best_threshold)

    y_pred = (y_prob >= best_threshold).astype(int)

    from sklearn.metrics import classification_report, roc_auc_score
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))



    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="Precision–Recall")
    plt.scatter(recall[best_idx], precision[best_idx], color="red", label="Chosen threshold")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve ({threshold_mode} mode)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(Path(OUTPUT_DIR) / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    return best, best_idx

def plot_probability_map(smashes, model, feature_cols, show_points=True, save_name=None):
    grid_x_m = np.linspace(0, COURT_WIDTH_M, 120)
    grid_y_m = np.linspace(0, COURT_LENGTH_M, 120)
    grid = pd.DataFrame([(gx, gy) for gx in grid_x_m for gy in grid_y_m], columns=["landing_x_m", "landing_y_m"])
    grid["player_location_x_m"] = smashes["player_location_x_m"].mean()
    grid["player_location_y_m"] = smashes["player_location_y_m"].mean()
    grid["distance_to_birdie_m"] = np.sqrt((grid["landing_x_m"] - grid["player_location_x_m"])**2 + (grid["landing_y_m"] - grid["player_location_y_m"])**2)
    grid["depth_m"] = grid["landing_y_m"]
    grid["width_m"] = grid["landing_x_m"]
    grid["to_center_m"] = np.abs(grid["landing_x_m"] - (COURT_WIDTH_M/2))
    grid["to_back_m"] = COURT_LENGTH_M - grid["landing_y_m"]
    grid["depth_x_width"] = grid["landing_y_m"] * grid["to_center_m"]
    grid["in_court_flag"] = 1
    Z = model.predict_proba(grid[feature_cols])[:, 1].reshape(len(grid_x_m), len(grid_y_m))

    aspect = COURT_LENGTH_M / COURT_WIDTH_M
    fig, ax = plt.subplots(figsize=(8, 8 * aspect), dpi=120)
    cs = ax.contourf(grid_x_m, grid_y_m, Z.T, levels=30, cmap="RdYlGn")
    plt.colorbar(cs, ax=ax, label="P(smash wins rally)")
    if show_points:
        success = smashes["smash_success"].astype(int).values
        ax.scatter(
            smashes.loc[success == 0, "landing_x_m"],
            smashes.loc[success == 0, "landing_y_m"],
            alpha=0.6, edgecolors="k", linewidths=0.5, label="0"
        )
        ax.scatter(
            smashes.loc[success == 1, "landing_x_m"],
            smashes.loc[success == 1, "landing_y_m"],
            alpha=0.6, edgecolors="k", linewidths=0.5, label="1"
        )

    draw_badminton_court(ax, mode="singles", width=COURT_WIDTH_M, length=COURT_LENGTH_M)
    ax.set_xlim(0, COURT_WIDTH_M)
    ax.set_ylim(0, COURT_LENGTH_M)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_name:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(Path(OUTPUT_DIR)/save_name, bbox_inches='tight')
    plt.show()

def save_smashes(smashes, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    smashes.to_csv(Path(output_dir)/"all_smashes_with_labels.csv", index=False, encoding="utf-8-sig")

def selftest_run():
    print("Running synthetic data test...")
    n = 300
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "type": rng.choice(["clear", "drop", "smash"], size=n, p=[0.2, 0.2, 0.6]),
        "landing_x": rng.normal(500, 150, size=n),
        "landing_y": rng.normal(800, 200, size=n),
        "player_location_x": rng.normal(300, 120, size=n),
        "player_location_y": rng.normal(200, 120, size=n),
        "player": rng.choice(["A", "B"], size=n),
        "getpoint_player": rng.choice(["A", "B"], size=n),
        "rally": rng.integers(1, 51, size=n),
        "ball_round": rng.integers(1, 10, size=n),
        "__match_name": rng.choice(["match1_MD", "match2_MS"], size=n),
        "__set_file": rng.choice(["set1.csv", "set2.csv"], size=n),
    })
    rally_w = get_rally_winners(df)
    smashes = get_all_smashes(df, rally_w)
    smashes, X, y, feature_cols = build_features_from_smashes(smashes)
    global smashes_global; smashes_global = smashes
    model, thr = train_smash_model_with_threshold(X, y, threshold_mode="precision", target_precision=0.7)
    save_smashes(smashes)
    plot_probability_map(smashes, model, feature_cols, show_points=True, save_name="selftest_with_dots.png")
    plot_probability_map(smashes, model, feature_cols, show_points=False, save_name="selftest_clean.png")
    print("Self-test complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--threshold-mode", choices=["precision", "f1"], default="precision")
    parser.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_PRECISION)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest_run()
        sys.exit(0)
    BASE_DIR = Path(__file__).resolve().parent
    SET_PATH = Path(args.data) if args.data else BASE_DIR/DEFAULT_SET_FOLDER
    full_df = load_all_sets(SET_PATH)
    rally_winners = get_rally_winners(full_df)
    smashes = get_all_smashes(full_df, rally_winners)
    smashes, X, y, feature_cols = build_features_from_smashes(smashes)
    global smashes_global; smashes_global = smashes
    model, prob_thr = train_smash_model_with_threshold(X, y, threshold_mode=args.threshold_mode, target_precision=args.target_precision)
    save_smashes(smashes)
    plot_probability_map(smashes, model, feature_cols, show_points=True, save_name="prob_map_with_dots.png")
    plot_probability_map(smashes, model, feature_cols, show_points=False, save_name="prob_map_clean.png")
    print
