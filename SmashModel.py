import pandas as pd
import numpy as np
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


SMASH_KEYWORDS = ['殺球', '點扣', 'smash']  # add more if your data uses other labels


def load_all_sets(base_path: Path):
    """
    Walk the dataset the SAME WAY your current code does:
    base_path/
        match_1/
            set1.csv
            set2.csv
        match_2/
            set1.csv
            ...
    Returns: list of (match_name, dataframe)
    """
    all_match_dfs = []
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

            # normalize columns to lowercase like your code
            df.columns = [c.strip().lower() for c in df.columns]
            df["__match_name"] = match_name
            df["__set_file"] = set_file.name
            all_match_dfs.append(df)
            print(f"  ✓ Loaded {set_file.name} ({len(df)} rows)")

    if not all_match_dfs:
        print("❌ No CSVs found under base path.")
        return pd.DataFrame()

    # concat everything
    full_df = pd.concat(all_match_dfs, ignore_index=True)
    return full_df


def get_last_shots(full_df: pd.DataFrame):
    """
    From the giant DF (all matches + sets), get the LAST shot of every rally,
    grouped by (match, set_file, rally).
    This is where we check: did the rally end with a smash, and who got the point?
    """
    required_cols = ["rally", "type", "player", "getpoint_player"]
    for col in required_cols:
        if col not in full_df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    # sort so tail(1) really gives the last chronological shot
    full_df = full_df.sort_values(
        ["__match_name", "__set_file", "rally", "ball_round"],
        ascending=True
    )

    last_shots = (
        full_df
        .groupby(["__match_name", "__set_file", "rally"])
        .tail(1)
        .copy()
    )

    return last_shots


def label_smash_success(last_shots: pd.DataFrame):
    """
    Add:
      - is_smash: did this rally end with a smash?
      - smash_success: was it a smash AND did the smasher win the point?
    """
    # some rows may not have 'type' as string
    last_shots["is_smash"] = last_shots["type"].astype(str).apply(
        lambda x: any(k.lower() in x.lower() for k in SMASH_KEYWORDS)
    )

    # success if: last shot was a smash and the hitter == point winner
    last_shots["smash_success"] = np.where(
        (last_shots["is_smash"]) &
        (last_shots["player"].astype(str) == last_shots["getpoint_player"].astype(str)),
        1,
        0
    )
    return last_shots


def build_features(smash_endings: pd.DataFrame):
    """
    Keep only rows where rally ended with smash, and build spatial features.
    """
    # keep only smashes
    smash_df = smash_endings[smash_endings["is_smash"]].copy()

    # some sets might miss coordinate columns, normalize names like in your extractor
    # try different possible names
    def get_col(row, main, fallback):
        return row.get(main) if main in row else row.get(fallback)

    # ensure cols exist
    needed = ["landing_x", "landing_y", "player_location_x", "player_location_y"]
    for col in needed:
        if col not in smash_df.columns:
            raise ValueError(f"Column '{col}' not found in your CSVs. Make sure all set files export coordinates.")

    # engineer distance
    smash_df["distance_to_birdie"] = np.sqrt(
        (smash_df["landing_x"] - smash_df["player_location_x"]) ** 2 +
        (smash_df["landing_y"] - smash_df["player_location_y"]) ** 2
    )

    # drop rows with missing coords
    smash_df = smash_df.dropna(
        subset=["landing_x", "landing_y", "player_location_x", "player_location_y"]
    )

    feature_cols = [
        "landing_x",
        "landing_y",
        "player_location_x",
        "player_location_y",
        "distance_to_birdie",
    ]

    X = smash_df[feature_cols]
    y = smash_df["smash_success"]

    return smash_df, X, y, feature_cols


def train_smash_model(X, y):
    """
    Train a simple logistic regression to predict
    P(smash wins rally | birdie + player location)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Smash Success Prediction Report ===")
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))

    return model


def save_processed_smashes(smash_df: pd.DataFrame, output_dir="smash_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "smash_rally_endings.csv"
    smash_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved processed smash rally endings to: {out_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    SET_PATH = BASE_DIR / "ShuttleSet/set"   # 👈 SAME as your original code

    if not SET_PATH.exists():
        print(f"⚠️ Default data folder not found at: {SET_PATH}")
        user_path = input("Enter the correct folder path for your badminton datasets: ").strip()
        if not user_path:
            print("❌ No valid path given. Exiting.")
            exit(1)
        SET_PATH = Path(user_path)

    print(f"Using data folder: {SET_PATH}")

    # 1. load everything like your original code
    full_df = load_all_sets(SET_PATH)
    if full_df.empty:
        print("❌ No data loaded. Exiting.")
        exit(1)

    # 2. get last shot in every rally
    last_shots = get_last_shots(full_df)

    # 3. label which rallies ended with smash + who won
    smash_endings = label_smash_success(last_shots)

    # 4. build features for modeling
    smash_df, X, y, feature_cols = build_features(smash_endings)

    print(f"\nTotal rallies: {len(last_shots)}")
    print(f"Rallies that ended with smash: {len(smash_df)}")
    print(f"Columns used for prediction: {feature_cols}")

    if len(smash_df) == 0:
        print("⚠️ No rallies ended with a smash. Check your keywords or your data.")
        exit(0)

    # 5. train model
    model = train_smash_model(X, y)

    # 6. save processed data (optional, but matches your style)
    save_processed_smashes(smash_df)
