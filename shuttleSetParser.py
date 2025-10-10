import pandas as pd
import numpy as np

def parse_smash_shots(csv_file_path):
    """
    Parse and extract all smash shot data from the badminton dataset
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing all smash shot data
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Filter only smash shots (殺球)
    smash_shots = df[df['type'] == '殺球'].copy()
    
    # Select relevant columns
    smash_data = smash_shots[[
        'rally', 'ball_round', 'time', 'frame_num', 'player',
        'hit_x', 'hit_y', 
        'landing_x', 'landing_y',
        'player_location_x', 'player_location_y',
        'opponent_location_x', 'opponent_location_y',
        'getpoint_player', 'lose_reason', 'win_reason'
    ]].copy()
    
    # Calculate additional metrics
    smash_data['smash_distance'] = np.sqrt(
        (smash_data['hit_x'] - smash_data['landing_x'])**2 + 
        (smash_data['hit_y'] - smash_data['landing_y'])**2
    )
    
    smash_data['player_to_hit_distance'] = np.sqrt(
        (smash_data['player_location_x'] - smash_data['hit_x'])**2 + 
        (smash_data['player_location_y'] - smash_data['hit_y'])**2
    )
    
    smash_data['smash_angle'] = np.degrees(np.arctan2(
        smash_data['landing_y'] - smash_data['hit_y'],
        smash_data['landing_x'] - smash_data['hit_x']
    ))
    
    # Add success indicator
    smash_data['success'] = smash_data['getpoint_player'] == smash_data['player']
    
    return smash_data

def get_smash_statistics(smash_data):
    """
    Calculate comprehensive statistics for smash shots
    
    Args:
        smash_data (pandas.DataFrame): DataFrame from parse_smash_shots
        
    Returns:
        dict: Dictionary containing smash statistics
    """
    
    stats = {
        'total_smashes': len(smash_data),
        'successful_smashes': smash_data['success'].sum(),
        'success_rate': smash_data['success'].mean() * 100,
        'avg_smash_distance': smash_data['smash_distance'].mean(),
        'avg_player_to_hit_distance': smash_data['player_to_hit_distance'].mean(),
        'avg_smash_angle': smash_data['smash_angle'].mean(),
        
        'by_player': {}
    }
    
    # Player-specific statistics
    for player in smash_data['player'].unique():
        player_data = smash_data[smash_data['player'] == player]
        stats['by_player'][player] = {
            'total_smashes': len(player_data),
            'successful_smashes': player_data['success'].sum(),
            'success_rate': player_data['success'].mean() * 100,
            'avg_smash_distance': player_data['smash_distance'].mean(),
            'avg_smash_angle': player_data['smash_angle'].mean(),
            'preferred_hit_locations': {
                'avg_hit_x': player_data['hit_x'].mean(),
                'avg_hit_y': player_data['hit_y'].mean()
            },
            'preferred_landing_locations': {
                'avg_landing_x': player_data['landing_x'].mean(),
                'avg_landing_y': player_data['landing_y'].mean()
            }
        }
    
    return stats

def visualize_smash_patterns(smash_data):
    """
    Generate basic visualization data for smash patterns
    
    Args:
        smash_data (pandas.DataFrame): DataFrame from parse_smash_shots
        
    Returns:
        dict: Visualization data
    """
    
    viz_data = {
        'hit_locations': smash_data[['hit_x', 'hit_y']].values.tolist(),
        'landing_locations': smash_data[['landing_x', 'landing_y']].values.tolist(),
        'player_locations': smash_data[['player_location_x', 'player_location_y']].values.tolist(),
        'successful_smashes': smash_data[smash_data['success']][['hit_x', 'hit_y', 'landing_x', 'landing_y']].values.tolist(),
        'failed_smashes': smash_data[~smash_data['success']][['hit_x', 'hit_y', 'landing_x', 'landing_y']].values.tolist()
    }
    
    return viz_data

# Example usage
if __name__ == "__main__":
    # Parse the smash shots
    smash_data = parse_smash_shots('set1.csv')
    
    print("=== SMASH SHOTS DATA ===")
    print(f"Total smash shots: {len(smash_data)}")
    print("\nFirst 5 smash shots:")
    print(smash_data.head())
    
    # Get statistics
    stats = get_smash_statistics(smash_data)
    
    print("\n=== SMASH STATISTICS ===")
    print(f"Total smashes: {stats['total_smashes']}")
    print(f"Successful smashes: {stats['successful_smashes']}")
    print(f"Success rate: {stats['success_rate']:.2f}%")
    print(f"Average smash distance: {stats['avg_smash_distance']:.2f}")
    print(f"Average player to hit distance: {stats['avg_player_to_hit_distance']:.2f}")
    print(f"Average smash angle: {stats['avg_smash_angle']:.2f}°")
    
    print("\n=== PLAYER-SPECIFIC STATISTICS ===")
    for player, player_stats in stats['by_player'].items():
        print(f"\nPlayer {player}:")
        print(f"  Total smashes: {player_stats['total_smashes']}")
        print(f"  Successful smashes: {player_stats['successful_smashes']}")
        print(f"  Success rate: {player_stats['success_rate']:.2f}%")
        print(f"  Average smash distance: {player_stats['avg_smash_distance']:.2f}")
        print(f"  Average smash angle: {player_stats['avg_smash_angle']:.2f}°")
        print(f"  Preferred hit location: ({player_stats['preferred_hit_locations']['avg_hit_x']:.1f}, {player_stats['preferred_hit_locations']['avg_hit_y']:.1f})")
        print(f"  Preferred landing location: ({player_stats['preferred_landing_locations']['avg_landing_x']:.1f}, {player_stats['preferred_landing_locations']['avg_landing_y']:.1f})")
    
    # Get visualization data
    viz_data = visualize_smash_patterns(smash_data)
    print(f"\n=== VISUALIZATION DATA ===")
    print(f"Hit locations: {len(viz_data['hit_locations'])} points")
    print(f"Landing locations: {len(viz_data['landing_locations'])} points")
    print(f"Successful smashes: {len(viz_data['successful_smashes'])}")
    print(f"Failed smashes: {len(viz_data['failed_smashes'])}")
    
    # Save to CSV for further analysis
    smash_data.to_csv('smash_analysis.csv', index=False)
    print("\nSmash data saved to 'smash_analysis.csv'")