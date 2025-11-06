import pandas as pd
from pathlib import Path
import os


def extract_all_smash_data(base_path):
    """
    Extract smash shot data (landing_x, landing_y, player_location_x, player_location_y)
    from all CSV files under the given dataset folder.
    """
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"❌ Error: Folder '{base_path}' not found.")
        print("Please make sure your dataset folder exists or specify the correct path.")
        return [], 0, 0

    all_smash_data = []
    processed_matches = 0
    total_smashes = 0

    print("Starting smash data extraction from all datasets...")

    for match_dir in base_path.iterdir():
        if match_dir.is_dir():
            print(f"\nProcessing match: {match_dir.name}")
            match_smashes = process_match_smashes(match_dir)
            if match_smashes:
                all_smash_data.extend(match_smashes)
                total_smashes += len(match_smashes)
            processed_matches += 1

            if processed_matches % 5 == 0:
                print(f"Processed {processed_matches} matches... Found {total_smashes} smashes so far")

    return all_smash_data, processed_matches, total_smashes


def process_match_smashes(match_dir):
    """
    Extract the desired columns from each set file containing smashes.
    """
    match_smashes = []
    smash_keywords = ['殺球', 'smash']  # Chinese + English keywords

    for set_file in match_dir.glob("*.csv"):
        try:
            df = pd.read_csv(set_file, encoding='utf-8-sig')

            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]

            if 'type' not in df.columns:
                print(f"  ⚠️ Skipping {set_file.name} - no 'type' column")
                continue

            # Filter only smash rows
            smash_mask = df['type'].astype(str).str.contains('|'.join(smash_keywords), case=False, na=False)
            smash_shots = df[smash_mask]

            if len(smash_shots) > 0:
                print(f"  ✓ {set_file.name}: {len(smash_shots)} smashes found")

            # Extract only the 4 needed columns
            for _, row in smash_shots.iterrows():
                smash_info = {
                    'match_name': match_dir.name,
                    'set_file': set_file.name,
                    'landing_x': row.get('landing_x', row.get('landingx', None)),
                    'landing_y': row.get('landing_y', row.get('landingy', None)),
                    'player_location_x': row.get('player_location_x', row.get('playerx', None)),
                    'player_location_y': row.get('player_location_y', row.get('playery', None))
                }
                match_smashes.append(smash_info)

        except Exception as e:
            print(f"  ✗ Error reading {set_file}: {e}")

    return match_smashes


def analyze_smash_patterns(smash_data):
    """
    Display simple summary of smash data.
    """
    if not smash_data:
        print("⚠️ No smash data found!")
        return None

    df = pd.DataFrame(smash_data)
    print("\n" + "=" * 60)
    print("SMASH COORDINATE SUMMARY")
    print("=" * 60)
    print(f"Total smash shots extracted: {len(df)}")
    print(f"Matches processed: {df['match_name'].nunique()}")
    print("\nSample data:")
    print(df.head())
    return df


def save_smash_data(smash_df, output_dir="smash_analysis"):
    """
    Save extracted smash data to CSV.
    """
    if smash_df is None or smash_df.empty:
        print("⚠️ No data to save.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "smash_coordinates.csv"
    smash_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved smash coordinate data to {output_file}")
    return output_file


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    SET_PATH = BASE_DIR / "set"

    if not SET_PATH.exists():
        print(f"⚠️ Default data folder not found at: {SET_PATH}")
        alt_path = input("Enter the correct folder path for your badminton datasets: ").strip()
        if alt_path:
            SET_PATH = Path(alt_path)
        else:
            print("❌ No valid path provided. Exiting program.")
            exit()

    print(f"Using data folder: {SET_PATH}")

    all_smash_data, processed_matches, total_smashes = extract_all_smash_data(SET_PATH)

    print(f"\nProcessed matches: {processed_matches}")
    print(f"Total smashes found: {total_smashes}")

    if all_smash_data:
        smash_df = analyze_smash_patterns(all_smash_data)
        save_smash_data(smash_df)
    else:
        print("\n❌ No smash data extracted. Check your 'set' folder path or file formats.")
