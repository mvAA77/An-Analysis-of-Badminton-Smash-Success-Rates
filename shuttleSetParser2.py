# bwf_research_simple.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math

class BWFResearchAnalyzer:
    def __init__(self, data_directory="ShuttleSet/set"):
        self.data_directory = data_directory
        self.combined_dataset = pd.DataFrame()
        
    def load_bwf_data_2018_2023(self, names_file="names.txt"):
        """Load BWF match-play and rally data from 2018-2023"""
        print("Loading BWF match-play and rally data (2018-2023)...")
        
        match_folders = self._load_match_folders(names_file)
        all_match_data = []
        
        print(f"Looking for match folders in: {self.data_directory}")
        print(f"Found {len(match_folders)} match folders in names.txt")
        
        for folder in match_folders:
            folder_path = os.path.join(self.data_directory, folder)
            print(f"Checking: {folder_path}")
            
            if os.path.exists(folder_path):
                match_df = self._process_match_folder(folder_path)
                if match_df is not None and not match_df.empty:
                    match_df['match_id'] = folder
                    # Extract year from folder name
                    year_part = folder.split('_')[-1]  # Get the last part which usually contains year
                    year = None
                    for part in folder.split('_'):
                        if part.isdigit() and len(part) == 4 and 2018 <= int(part) <= 2023:
                            year = int(part)
                            break
                    
                    if year is None:
                        # Try to extract from tournament name
                        if '2018' in folder:
                            year = 2018
                        elif '2019' in folder:
                            year = 2019
                        elif '2020' in folder:
                            year = 2020
                        elif '2021' in folder:
                            year = 2021
                        else:
                            year = 2022  # Default for recent tournaments
                    
                    match_df['year'] = year
                    all_match_data.append(match_df)
                    print(f"  ✓ Loaded data from {folder}")
                else:
                    print(f"  ✗ No valid data in {folder}")
            else:
                print(f"  ✗ Folder not found: {folder_path}")
        
        if all_match_data:
            self.combined_dataset = pd.concat(all_match_data, ignore_index=True)
            print(f"\n✓ BWF Dataset (2018-2023): {len(self.combined_dataset)} shots from {len(all_match_data)} matches")
            
            # Show year distribution
            year_counts = self.combined_dataset['year'].value_counts().sort_index()
            print("Year distribution:", dict(year_counts))
        else:
            print("\n✗ No BWF match data found for 2018-2023")
            print("Please check:")
            print("1. The data_directory path is correct")
            print("2. The folder names in names.txt match actual folder names")
            print("3. CSV files exist in the match folders")
            
        return self.combined_dataset
    
    def _load_match_folders(self, names_file):
        """Load match folders from names file"""
        try:
            with open(names_file, 'r', encoding='utf-8') as file:
                folders = [line.strip() for line in file if line.strip()]
                print(f"Read {len(folders)} folders from {names_file}")
                return folders
        except FileNotFoundError:
            print(f"✗ Names file {names_file} not found")
            print("Creating sample names file...")
            self._create_sample_names_file()
            return []
    
    def _create_sample_names_file(self):
        """Create a sample names file if it doesn't exist"""
        sample_matches = [
            "An_Se_Young_Pornpawee_Chochuwong_TOYOTA_THAILAND_OPEN_2021_QuarterFinals",
            "An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals",
            "Anthony_Sinisuka_Ginting_Lee_Zii_Jia_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals"
        ]
        
        with open("names.txt", "w", encoding="utf-8") as f:
            for match in sample_matches:
                f.write(match + "\n")
        
        print("✓ Created sample names.txt file")
        print("Please update it with your actual match folder names")
    
    def _process_match_folder(self, folder_path):
        """Process BWF match data"""
        csv_files = list(Path(folder_path).glob('*.csv'))
        
        if not csv_files:
            print(f"    No CSV files found in {folder_path}")
            return None
        
        print(f"    Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        
        match_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"      ✓ Loaded {csv_file.name}: {len(df)} rows")
                df = self._standardize_bwf_data(df)
                match_data.append(df)
            except Exception as e:
                print(f"      ✗ Error reading {csv_file}: {e}")
        
        if match_data:
            combined_df = pd.concat(match_data, ignore_index=True)
            print(f"      Combined: {len(combined_df)} shots")
            return combined_df
        return None
    
    def _standardize_bwf_data(self, df):
        """Standardize BWF data format"""
        # Create rally outcome variable
        if 'win_reason' in df.columns:
            df['won_rally'] = df['win_reason'].notna() & ~df['win_reason'].str.contains('FAULT|Fault', na=False)
        else:
            df['won_rally'] = False
            
        # Standardize shot categories
        if 'type' in df.columns:
            df['shot_category'] = df['type'].astype(str).str.strip()
            
        return df
    
    def analyze_shot_probabilities(self):
        """
        RESEARCH OBJECTIVE 1: Analyze if certain shots have higher probabilities of winning
        This prepares data for your existing Chi-Square test in calculator.py
        """
        if self.combined_dataset.empty:
            print("No data available for analysis")
            return None
        
        print("\n" + "="*70)
        print("SHOT WIN PROBABILITY ANALYSIS")
        print("="*70)
        print("Analyzing if certain shots have higher probabilities of winning rallies")
        print("-" * 70)
        
        # Get shot categories with sufficient data
        shot_counts = self.combined_dataset['shot_category'].value_counts()
        print(f"Total shot types found: {len(shot_counts)}")
        
        valid_shots = shot_counts[shot_counts >= 5].index  # Reduced minimum for more data
        
        if len(valid_shots) < 2:
            print("Insufficient shot categories for analysis")
            print("Available shots:", dict(shot_counts.head(10)))
            return None
        
        # Calculate win probabilities
        winning_shots = self.combined_dataset[self.combined_dataset['won_rally']]
        win_probabilities = []
        
        print(f"\nAnalyzing {len(valid_shots)} shot categories...")
        
        for shot in valid_shots:
            total = shot_counts[shot]
            wins = len(winning_shots[winning_shots['shot_category'] == shot])
            win_prob = (wins / total) * 100
            win_probabilities.append((shot, win_prob, wins, total))
        
        # Sort by win probability
        win_probabilities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nWIN PROBABILITIES BY SHOT CATEGORY:")
        print("-" * 60)
        for shot, prob, wins, total in win_probabilities:
            print(f"{shot:15} {prob:6.1f}% ({wins:3d}/{total:3d})")
        
        # Prepare data for Chi-Square test (to use with your calculator.py)
        shot_names = [item[0] for item in win_probabilities]
        shot_counts_list = [item[3] for item in win_probabilities]
        
        print(f"\nDATA FOR CHI-SQUARE TEST (use in calculator.py):")
        print(f"Shot types: {shot_names}")
        print(f"Counts: {shot_counts_list}")
        
        self._plot_shot_probabilities(win_probabilities)
        
        return {
            'win_probabilities': win_probabilities,
            'shot_names': shot_names,
            'shot_counts': shot_counts_list
        }
    
    def _plot_shot_probabilities(self, win_probabilities):
        """Visualize shot win probabilities"""
        plt.figure(figsize=(12, 6))
        
        shots = [item[0] for item in win_probabilities[:15]]  # Top 15
        probs = [item[1] for item in win_probabilities[:15]]
        
        # Simple color coding
        avg_prob = np.mean(probs)
        colors = ['red' if prob > avg_prob else 'blue' for prob in probs]
        
        bars = plt.bar(shots, probs, color=colors, alpha=0.7)
        
        plt.axhline(y=avg_prob, color='red', linestyle='--', 
                   label=f'Average: {avg_prob:.1f}%')
        
        plt.title('Win Probability by Shot Category\n(Use with Chi-Square Test for Homogeneity)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Shot Category')
        plt.ylabel('Win Probability (%)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('shot_win_probabilities.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_smash_success_locations(self):
        """
        RESEARCH OBJECTIVE 2: Analyze smash success based on locations
        """
        if self.combined_dataset.empty:
            print("No data available for smash analysis")
            return None
        
        print("\n" + "="*70)
        print("SMASH SUCCESS LOCATION ANALYSIS")
        print("="*70)
        print("Analyzing smash success based on birdie and racket locations")
        print("-" * 70)
        
        # Filter for smash shots
        smash_keywords = ['殺球', 'Smash', 'smash', '扣杀']
        smash_mask = False
        for keyword in smash_keywords:
            smash_mask = smash_mask | self.combined_dataset['shot_category'].str.contains(keyword, na=False)
        
        smash_data = self.combined_dataset[smash_mask].copy()
        
        if smash_data.empty:
            print("No smash data found in the dataset")
            print("Available shot types:", self.combined_dataset['shot_category'].unique()[:10])
            return None
        
        print(f"Analyzing {len(smash_data)} smash shots...")
        
        # Overall smash statistics
        total_smashes = len(smash_data)
        successful_smashes = smash_data['won_rally'].sum()
        success_rate = (successful_smashes / total_smashes) * 100
        
        print(f"\nOVERALL SMASH STATISTICS:")
        print(f"Total smashes: {total_smashes}")
        print(f"Successful smashes: {successful_smashes}")
        print(f"Success rate: {success_rate:.1f}%")
        
        # Location analysis
        location_analysis = {}
        
        # Analyze by hit area if available
        if 'hit_area' in smash_data.columns:
            hit_area_success = smash_data.groupby('hit_area')['won_rally'].agg(['mean', 'count'])
            hit_area_success = hit_area_success[hit_area_success['count'] >= 3]  # Reduced minimum
            hit_area_success = hit_area_success.sort_values('mean', ascending=False)
            
            if not hit_area_success.empty:
                print(f"\nSMASH SUCCESS BY HIT AREA (Top 5):")
                for area, (success_rate, count) in hit_area_success.head(5).iterrows():
                    print(f"  Area {area}: {success_rate*100:.1f}% ({int(count)} shots)")
                location_analysis['hit_area'] = hit_area_success
        
        # Analyze by landing area if available
        if 'landing_area' in smash_data.columns:
            landing_area_success = smash_data.groupby('landing_area')['won_rally'].agg(['mean', 'count'])
            landing_area_success = landing_area_success[landing_area_success['count'] >= 3]
            landing_area_success = landing_area_success.sort_values('mean', ascending=False)
            
            if not landing_area_success.empty:
                print(f"\nSMASH SUCCESS BY LANDING AREA (Top 5):")
                for area, (success_rate, count) in landing_area_success.head(5).iterrows():
                    print(f"  Area {area}: {success_rate*100:.1f}% ({int(count)} shots)")
                location_analysis['landing_area'] = landing_area_success
        
        self._plot_smash_analysis(smash_data, success_rate, location_analysis)
        
        return {
            'total_smashes': total_smashes,
            'successful_smashes': successful_smashes,
            'success_rate': success_rate,
            'location_analysis': location_analysis
        }
    
    def _plot_smash_analysis(self, smash_data, success_rate, location_analysis):
        """Visualize smash analysis results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Success rate
        labels = ['Successful', 'Failed']
        sizes = [success_rate, 100 - success_rate]
        colors = ['lightgreen', 'lightcoral']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Smash Success Rate')
        
        # Plot 2: Location effectiveness or shot distribution
        if location_analysis.get('hit_area'):
            hit_areas = location_analysis['hit_area'].head(5)
            areas = [str(area) for area in hit_areas.index]
            success_rates = hit_areas['mean'] * 100
            
            bars = ax2.bar(areas, success_rates, color='skyblue', alpha=0.7)
            ax2.set_title('Smash Success by Hit Area (Top 5)')
            ax2.set_xlabel('Hit Area')
            ax2.set_ylabel('Success Rate (%)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, rate in zip(bars, success_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            # Show shot distribution instead
            shot_types = smash_data['shot_category'].value_counts().head(5)
            ax2.bar([str(x) for x in shot_types.index], shot_types.values, color='orange', alpha=0.7)
            ax2.set_title('Most Common Smash Types')
            ax2.set_xlabel('Smash Type')
            ax2.set_ylabel('Number of Shots')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('smash_location_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_strategic_recommendations(self, shot_analysis, smash_analysis):
        """
        RESEARCH OBJECTIVE 3: Provide insights for players and coaches
        """
        print("\n" + "="*70)
        print("STRATEGIC RECOMMENDATIONS")
        print("="*70)
        print("Data-driven insights for offensive and defensive strategies")
        print("-" * 70)
        
        if shot_analysis:
            print("\n🎯 OFFENSIVE STRATEGIES:")
            top_shots = shot_analysis['win_probabilities'][:3]
            print("Based on win probability analysis, focus on:")
            for shot, prob, wins, total in top_shots:
                print(f"  • {shot} shots ({prob:.1f}% success rate)")
            print("  • Develop sequences that create opportunities for these shots")
            
            print("\n🛡️ DEFENSIVE STRATEGIES:")
            bottom_shots = shot_analysis['win_probabilities'][-3:]
            print("When defending, try to force opponents into:")
            for shot, prob, wins, total in bottom_shots:
                print(f"  • {shot} shots ({prob:.1f}% success rate)")
        
        if smash_analysis:
            print("\n💥 SMASH-SPECIFIC INSIGHTS:")
            print(f"Overall smash success: {smash_analysis['success_rate']:.1f}%")
            if smash_analysis['location_analysis'].get('hit_area') is not None:
                best_area = smash_analysis['location_analysis']['hit_area'].iloc[0]
                print(f"  • Most effective hit area: {best_area.name} ({best_area['mean']*100:.1f}% success)")
        
        print(f"\n📊 NEXT STEPS FOR RESEARCH:")
        print("1. Use the shot counts above in your calculator.py for Chi-Square test")
        print("2. The Chi-Square test will statistically verify if win probabilities differ")
        print("3. Apply location-based insights to training drills")
    
    def run_complete_analysis(self):
        """Execute the complete research pipeline"""
        print("BADMINTON PERFORMANCE ANALYSIS")
        print("Using BWF match-play data 2018-2023")
        print("="*70)
        
        # Load data
        self.load_bwf_data_2018_2023()
        
        if self.combined_dataset.empty:
            print("No data available for analysis")
            return
        
        # Analyze shot probabilities
        shot_analysis = self.analyze_shot_probabilities()
        
        # Analyze smash success
        smash_analysis = self.analyze_smash_success_locations()
        
        # Generate recommendations
        self.generate_strategic_recommendations(shot_analysis, smash_analysis)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)


def main():
    """Main execution function with better error handling"""
    # Try different possible directory structures
    possible_paths = [
        "ShuttleSet/set",
        "Shuttleset/set", 
        "ShuttleSet",
        "Shuttleset",
        ".",
        "data",
        "dataset"
    ]
    
    analyzer = None
    for path in possible_paths:
        test_analyzer = BWFResearchAnalyzer(data_directory=path)
        print(f"Trying path: {path}")
        
        # Quick test to see if this path works
        test_path = Path(path)
        if test_path.exists():
            print(f"✓ Path exists: {path}")
            analyzer = test_analyzer
            break
        else:
            print(f"✗ Path not found: {path}")
    
    if analyzer is None:
        print("No valid data directory found. Please check your file structure.")
        return
    
    # Run debugging
    analyzer.debug_directory_structure()
    
    # Then run analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()