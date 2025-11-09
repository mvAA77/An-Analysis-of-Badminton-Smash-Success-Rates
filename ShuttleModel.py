import os
from pathlib import Path

import pandas as pd
import numpy as np

# make sure these are installed in your venv:
#   pip install scikit-learn matplotlib seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches


# CONSTANTS / CONFIG
# keywords that mean "smash" in data
SMASH_KEYWORDS = ["殺球", "點扣", "smash"]

DEFAULT_SET_FOLDER = "ShuttleSet/set"

OUTPUT_DIR = "smash_analysis"

# court dimensions from the ShuttleSet
COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4


# HELPERS: match mode, in/out, court drawing
def infer_match_mode_from_name(match_name: str) -> str:
    """
    Infer singles vs doubles from the match/folder name string.
    This version works with Series.apply(...) on a single column.
    """
    name_lower = str(match_name).lower()
    doubles_keywords = ["md", "wd", "xd", "double", "doubles", "mixed"]
    if any(k in name_lower for k in doubles_keywords):
        return "doubles"
    return "singles"



def is_in_rally_court(landing_x_m: float, landing_y_m: float, mode: str) -> bool:
    """
    Check IN/OUT for *rally* shots (not serve).
    Singles: inner sidelines (0.46m in from each side), full length 13.4
    Doubles: full width 6.1m, full length 13.4
    """
    if mode == "doubles":
        return (0.0 <= landing_x_m <= COURT_WIDTH_M) and (0.0 <= landing_y_m <= COURT_LENGTH_M)
    else:
        # singles: playable area is between 0.46 and 6.1 - 0.46
        left = 0.46
        right = COURT_WIDTH_M - 0.46  # 6.1 - 0.46 = 5.64 → width = 5.18
        return (left <= landing_x_m <= right) and (0.0 <= landing_y_m <= COURT_LENGTH_M)


def draw_badminton_court(ax, mode="singles", width=6.1, length=13.4):
    """
    Draws a badminton court on the given axes.
    mode = 'singles' -> draw inner sidelines
    mode = 'doubles' -> draw full width
    """
    # outer boundary
    outer = patches.Rectangle(
        (0, 0), width, length,
        linewidth=2,
        edgecolor="black",
        facecolor="none"
    )
    ax.add_patch(outer)

    # net line at 6.7m
    net_y = length / 2
    ax.axhline(net_y, color="black", linestyle="--", linewidth=2)

    # short service lines (1.98m from net, both sides)
    ax.hlines([net_y - 1.98, net_y + 1.98], 0, width, colors="gray", linestyles="-", linewidth=1.2)

    # center service line
    ax.vlines(width / 2, net_y - 1.98, net_y + 1.98, colors="gray", linestyles="-", linewidth=1.2)

    if mode == "singles":
        # singles sidelines 0.46m in from each side
        ax.vlines([0.46, width - 0.46], 0, length, colors="gray", linestyles="-", linewidth=1.2)
        # long service line for singles (for reference, 0.76m from back)
        ax.hlines([0.76, length - 0.76], 0.46, width - 0.46, colors="gray", linestyles="-", linewidth=1.0)
    else:
        # doubles: full width; show long service line for doubles serve
        ax.hlines([0.76, length - 0.76], 0, width, colors="gray", linestyles=":", linewidth=1.0)

    # labels
    ax.text(width / 2, net_y + 0.25, "Opponent Side", ha="center", fontsize=9)
    ax.text(width / 2, net_y - 0.6, "Player Side", ha="center", fontsize=9)


# 1 LOAD ALL CSVs LIKE YOUR ORIGINAL CODE
def load_all_sets(base_path: Path) -> pd.DataFrame:
    all_match_dfs = []

    if not base_path.exists():
        print(f"❌ Error: folder '{base_path}' not found.")
        return pd.DataFrame()

    print("📦 Loading all match/set CSVs...")

    for match_dir in base_path.iterdir():
        if not match_dir.is_dir():
            continue

        match_name = match_dir.name
        print(f"\n📁 Match: {match_name}")

        for set_file in match_dir.glob("*.csv"):
            try:
                df = pd.read_csv(set_file, encoding="utf-8-sig")
            except Exception as e:
                print(f"  ✗ Error reading {set_file}: {e}")
                continue

            df.columns = [c.strip().lower() for c in df.columns]
            df["__match_name"] = match_name
            df["__set_file"] = set_file.name
            all_match_dfs.append(df)
            print(f"  ✓ {set_file.name}: {len(df)} rows")

    if not all_match_dfs:
        print("⚠️ No CSVs found.")
        return pd.DataFrame()

    full_df = pd.concat(all_match_dfs, ignore_index=True)
    print(f"\n✅ Finished loading. Total rows: {len(full_df)}")
    return full_df


# 2 GET RALLY WINNER (last row of each rally)
def get_rally_winners(full_df: pd.DataFrame) -> pd.DataFrame:
    required = ["rally", "getpoint_player", "ball_round"]
    for c in required:
        if c not in full_df.columns:
            raise ValueError(f"Dataset missing required column: {c}")

    full_df = full_df.sort_values(
        ["__match_name", "__set_file", "rally", "ball_round"],
        ascending=True
    )

    rally_winners = (
        full_df
        .groupby(["__match_name", "__set_file", "rally"])
        .tail(1)[["__match_name", "__set_file", "rally", "getpoint_player"]]
        .rename(columns={"getpoint_player": "rally_winner"})
        .copy()
    )
    return rally_winners

# 3 GET ALL SMASHES (not only last shot)
def get_all_smashes(full_df: pd.DataFrame, rally_winners: pd.DataFrame) -> pd.DataFrame:
    if "type" not in full_df.columns:
        raise ValueError("CSV missing 'type' column.")

    # filter all smash rows
    smash_mask = full_df["type"].astype(str).apply(
        lambda x: any(k.lower() in x.lower() for k in SMASH_KEYWORDS)
    )
    smashes = full_df[smash_mask].copy()
    print(f"✅ Found {len(smashes)} smashes in all rallies")

    # attach rally winner to each smash
    smashes = smashes.merge(
        rally_winners,
        on=["__match_name", "__set_file", "rally"],
        how="left",
        validate="m:1"
    )

    # label success: did the smasher's side win?
    smashes["smash_success"] = np.where(
        smashes["player"].astype(str) == smashes["rally_winner"].astype(str),
        1,
        0
    )

    return smashes


# 4 BUILD FEATURES FROM SMASHES
def build_features_from_smashes(smashes: pd.DataFrame):
    needed = ["landing_x", "landing_y", "player_location_x", "player_location_y"]
    for col in needed:
        if col not in smashes.columns:
            raise ValueError(
                f"Column '{col}' not found in your CSVs. "
                f"Make sure all set files export coordinates."
            )

    smashes = smashes.dropna(subset=needed).copy()

    # distance feature (still pixels)
    smashes["distance_to_birdie"] = np.sqrt(
        (smashes["landing_x"] - smashes["player_location_x"]) ** 2 +
        (smashes["landing_y"] - smashes["player_location_y"]) ** 2
    )

    feature_cols = [
        "landing_x",
        "landing_y",
        "player_location_x",
        "player_location_y",
        "distance_to_birdie",
    ]

    X = smashes[feature_cols]
    y = smashes["smash_success"]

    print(f"✅ Built features for {len(smashes)} smashes")
    return smashes, X, y, feature_cols


# 5 TRAIN MODEL
def train_smash_model(X, y):
    if len(X) < 20:
        print("⚠️ Not enough smashes to train a model.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Smash Success Prediction Report (all smashes) ===")
    print(classification_report(y_test, y_pred, digits=3))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print("AUC:", round(auc, 3))
    except ValueError:
        print("AUC could not be computed (only one class).")

    return model


# 6 PLOT PROBABILITY MAP (with singles/doubles + full court)
def plot_probability_map(smashes: pd.DataFrame, model, feature_cols):
    if model is None:
        print("⚠️ Model is None, skipping probability map.")
        return

    # infer mode (singles/doubles) per row
    smashes = smashes.copy()
    smashes["match_mode"] = smashes["__match_name"].apply(infer_match_mode_from_name)

    # get pixel ranges from your data
    px_x_min = smashes["landing_x"].min()
    px_x_max = smashes["landing_x"].max()
    px_y_min = smashes["landing_y"].min()
    px_y_max = smashes["landing_y"].max()

    if px_x_min == px_x_max or px_y_min == px_y_max:
        print("⚠️ Not enough variation to build heatmap.")
        return

    # pixel: meter scale
    x_range_px = px_x_max - px_x_min
    y_range_px = px_y_max - px_y_min
    x_scale = COURT_WIDTH_M / x_range_px
    y_scale = COURT_LENGTH_M / y_range_px

    # build grid in PIXELS
    grid_x_px = np.linspace(px_x_min, px_x_max, 60)
    grid_y_px = np.linspace(px_y_min, px_y_max, 60)
    pts = [(gx, gy) for gx in grid_x_px for gy in grid_y_px]
    grid = pd.DataFrame(pts, columns=["landing_x", "landing_y"])

    # average player pos (pixels)
    avg_player_x_px = smashes["player_location_x"].mean()
    avg_player_y_px = smashes["player_location_y"].mean()
    grid["player_location_x"] = avg_player_x_px
    grid["player_location_y"] = avg_player_y_px
    grid["distance_to_birdie"] = np.sqrt(
        (grid["landing_x"] - grid["player_location_x"]) ** 2 +
        (grid["landing_y"] - grid["player_location_y"]) ** 2
    )

    # predict in pixel space
    grid["pred_prob"] = model.predict_proba(grid[feature_cols])[:, 1]

    # convert to meters for plotting
    grid["landing_x_m"] = (grid["landing_x"] - px_x_min) * x_scale
    grid["landing_y_m"] = (grid["landing_y"] - px_y_min) * y_scale

    smashes_plot = smashes.copy()
    smashes_plot["landing_x_m"] = (smashes_plot["landing_x"] - px_x_min) * x_scale
    smashes_plot["landing_y_m"] = (smashes_plot["landing_y"] - px_y_min) * y_scale

    # mark which smashes are actually in for *their* mode
    smashes_plot["in_court"] = smashes_plot.apply(
        lambda r: is_in_rally_court(r["landing_x_m"], r["landing_y_m"], r["match_mode"]),
        axis=1
    )

    # reshape prob grid
    Z = grid["pred_prob"].values.reshape(len(grid_x_px), len(grid_y_px))
    grid_x_m = (grid_x_px - px_x_min) * x_scale
    grid_y_m = (grid_y_px - px_y_min) * y_scale

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # heatmap
    cs = ax.contourf(grid_x_m, grid_y_m, Z.T, levels=20, cmap="RdYlGn")
    plt.colorbar(cs, label="P(smash wins rally)")

    # in-court smashes
    sns.scatterplot(
        data=smashes_plot[smashes_plot["in_court"]],
        x="landing_x_m",
        y="landing_y_m",
        hue="smash_success",
        palette="coolwarm",
        alpha=0.6,
        edgecolor="k",
        legend=False,
        ax=ax
    )

    # out-of-court-for-that-mode smashes
    out_df = smashes_plot[~smashes_plot["in_court"]]
    if not out_df.empty:
        ax.scatter(
            out_df["landing_x_m"],
            out_df["landing_y_m"],
            marker="x",
            color="red",
            label="Out (by mode)",
            alpha=0.9
        )

    # draw singles court overlay (most matches will be singles)
    draw_badminton_court(ax, mode="singles", width=COURT_WIDTH_M, length=COURT_LENGTH_M)

    ax.set_xlim(0, COURT_WIDTH_M)
    ax.set_ylim(0, COURT_LENGTH_M)
    ax.set_title("Smash Win Probability (mode-aware, with court overlay)")
    ax.set_xlabel("Court width (m)")
    ax.set_ylabel("Court length / depth (m)")
    plt.tight_layout()
    plt.show()


# 7 SAVE LABELED SMASHES
def save_smashes(smashes: pd.DataFrame, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "all_smashes_with_labels.csv"
    smashes.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved labeled smashes to: {out_path}")


# MAIN
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    SET_PATH = BASE_DIR / DEFAULT_SET_FOLDER

    if not SET_PATH.exists():
        print(f"⚠️ Default data folder not found at: {SET_PATH}")
        alt_path = input("Enter the correct folder path for your badminton datasets: ").strip()
        if not alt_path:
            print("❌ No valid path provided. Exiting.")
            exit(1)
        SET_PATH = Path(alt_path)

    print(f"📂 Using data folder: {SET_PATH}")

    # 1. load everything
    full_df = load_all_sets(SET_PATH)
    if full_df.empty:
        print("❌ No data loaded. Exiting.")
        exit(1)

    # 2. rally winners
    rally_winners = get_rally_winners(full_df)

    # 3. all smashes (anywhere in rally)
    smashes = get_all_smashes(full_df, rally_winners)

    # 4. build features
    smashes, X, y, feature_cols = build_features_from_smashes(smashes)

    print(f"\nTotal smashes (all): {len(smashes)}")

    # 5. train model
    model = train_smash_model(X, y)

    # 6. save data
    save_smashes(smashes, OUTPUT_DIR)

    # 7. plot
    plot_probability_map(smashes, model, feature_cols)