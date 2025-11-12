"""
ChiSquareAnalysis_English.py
----------------------------
Performs a Chi-Square test between SHOT TYPE and RALLY OUTCOME across all matches.

Enhancements:
- Translates Chinese shot types to English (including newly added terms).
- Ignores 'fault' and out-of-play shots.
- Uses only the rally-ending shot.
- Saves an English contingency table (CSV).
- Generates stacked bar + win-rate charts.
"""

from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

OUTPUT_DIR = "chi_square_output"

# ----------------- TRANSLATION TABLE -----------------
SHOT_TYPE_TRANSLATION = {
    # Core types
    "殺球": "Smash",
    "點扣": "Smash",
    "殺球未中": "Missed Smash",
    "過度切球": "Half Smash / Cut",
    "平球": "Drive",
    "抽球": "Clear",
    "後場抽平球": "Flat Clear",
    "挑球": "Lift",
    "挑後場球": "Deep Lift",
    "防守挑球": "Defensive Lift",
    "切球": "Drop",
    "放小球": "Net Shot",
    "推球": "Push",
    "擋小球": "Block",
    "勾球": "Net Kill",
    "發長球": "Long Serve",
    "發短球": "Short Serve",
    "發球": "Serve",
    "小平球": "Flat Drive",            # newly added
    "撲球": "Tap / Pounce Shot",      # newly added
    "未知球種": "Unknown Shot Type",   # newly added
    "長球": "High Clear / Long Lift",  # newly added
    "防守回抽": "Defensive Clear",     # newly added
    "防守回挑": "Defensive Lift (Return)",  # newly added
    # Common outcome/error labels
    "掛網": "Net Fault",
    "出界": "Out"
}

FAULT_KEYWORDS = [
    "fault", "let", "失誤", "違例", "違規", "掛網", "出界"
]

# ----------------- HELPERS -----------------
def translate_shot_type(s: str) -> str:
    if not isinstance(s, str):
        return "Unknown"
    s_clean = s.strip()
    return SHOT_TYPE_TRANSLATION.get(s_clean, s_clean)

def is_fault_type(s: str) -> bool:
    """Return True if this shot is a known fault/out/violation."""
    if not isinstance(s, str):
        return False
    s_low = s.lower()
    if any(k in s_low for k in FAULT_KEYWORDS):
        return True
    eng = translate_shot_type(s)
    return eng in {"Net Fault", "Out"}

def load_final_shots(data_root: Path) -> pd.DataFrame:
    """Load all matches, keep final rally shots, ignore faults, translate to English."""
    all_rows = []
    for match_dir in data_root.iterdir():
        if not match_dir.is_dir():
            continue
        for set_file in match_dir.glob("*.csv"):
            try:
                df = pd.read_csv(set_file, encoding="utf-8-sig")
                df.columns = [c.strip().lower() for c in df.columns]

                req = {"type", "player", "getpoint_player"}
                if not req.issubset(df.columns):
                    continue

                # Keep final shot of each rally
                if "rally" in df.columns:
                    df["rally"] = pd.to_numeric(df["rally"], errors="coerce")
                    df["__match"] = match_dir.name
                    df["__set"] = set_file.name
                    df_end = (
                        df.sort_values(["__match", "__set", "rally"])
                          .groupby(["__match", "__set", "rally"], dropna=True, as_index=False)
                          .tail(1)
                    )
                else:
                    df_end = df.copy()

                # Remove faults
                df_end = df_end[df_end["type"].notna()]
                df_end = df_end[~df_end["type"].apply(is_fault_type)]

                # Translate to English
                df_end["type_en"] = df_end["type"].apply(translate_shot_type)

                # Binary win flag
                df_end["rally_winner"] = (df_end["player"] == df_end["getpoint_player"]).astype(int)

                all_rows.append(df_end[["__match", "__set", "type_en", "rally_winner"]])

            except Exception as e:
                print(f"⚠️ Error reading {set_file}: {e}")

    if not all_rows:
        return pd.DataFrame(columns=["__match", "__set", "type_en", "rally_winner"])
    return pd.concat(all_rows, ignore_index=True)

# ----------------- VISUALS -----------------
def plot_stacked(contingency: pd.DataFrame, out_dir: Path):
    ax = contingency.plot(kind="bar", stacked=True, figsize=(10,6),
                          title="Rally Outcome Counts by Shot Type (Final Shot Only)")
    ax.set_xlabel("Shot Type (English)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = out_dir / "stacked_counts_by_shot_type.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved stacked count chart: {path}")

def plot_winrate(contingency: pd.DataFrame, out_dir: Path):
    rate = contingency.div(contingency.sum(axis=1), axis=0)["Won Rally (1)"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,6))
    rate.plot(kind="bar", ax=ax, color="green", title="Win Rate by Shot Type (Final Shot Only)")
    ax.set_xlabel("Shot Type (English)")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0,1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x()+p.get_width()/2, p.get_height()+0.01),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = out_dir / "win_rate_by_shot_type.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Saved win rate chart: {path}")

# ----------------- MAIN -----------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "ShuttleSet\set"
    if not data_dir.exists():
        print(f"⚠️ Dataset folder not found at {data_dir}")
        alt = input("Enter dataset path: ").strip()
        data_dir = Path(alt)

    print("🔍 Loading data...")
    df = load_final_shots(data_dir)
    if df.empty:
        print("❌ No usable data found.")
        exit()

    contingency = pd.crosstab(df["type_en"], df["rally_winner"])
    contingency.columns = ["Lost Rally (0)", "Won Rally (1)"]

    out_dir = base_dir / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "shot_type_contingency_english.csv"
    contingency.to_csv(csv_path, encoding="utf-8-sig")
    print("\n===============================")
    print("SHOT TYPE CONTINGENCY TABLE (ENGLISH)")
    print("===============================")
    print(contingency)
    print(f"\n📁 Saved contingency table: {csv_path}")

    # Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\n=== CHI-SQUARE TEST RESULTS ===")
    print(f"Chi2 Statistic: {chi2:.3f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p:.6f}")
    if p < 0.05:
        print("\n✅ Reject H₀: Shot type and rally success are NOT independent.")
        print("→ Some shot types are significantly more likely to win rallies.")
    else:
        print("\n❌ Fail to reject H₀: No significant difference found.")
        print("→ All shot types appear equally likely to win rallies.")

    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    expected_df.to_csv(out_dir / "expected_counts_english.csv", encoding="utf-8-sig")
    print(f"🧮 Saved expected counts CSV.")

    # Charts
    plot_stacked(contingency, out_dir)
    plot_winrate(contingency, out_dir)
    print("\n✅ Analysis complete. Results saved in:", out_dir.resolve())
