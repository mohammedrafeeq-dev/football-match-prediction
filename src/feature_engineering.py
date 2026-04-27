import pandas as pd
import numpy as np

def calculate_rolling_averages(group, cols, new_cols):
    """
    Calculates rolling averages for a team's performance
    Shift(1) is used to avoid data leakage (don't include current match stats)
    """
    group = group.sort_values('Date')
    rolling_stats = group[cols].rolling(5, closed='left').mean()
    group[new_cols] = rolling_stats
    # Drop rows where rolling stats are not available (first few matches)
    return group.dropna(subset=new_cols)

def get_form_points(result):
    if result == 'H': return 3
    if result == 'D': return 1
    return 0

def calculate_team_form(df):
    """
    Calculates team form (points in last 5 games)
    """
    # Create two entries for each match: one for home team, one for away team
    home_matches = df[['Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team', 'FTR': 'Result'})
    home_matches['Points'] = home_matches['Result'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    
    away_matches = df[['Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team', 'FTR': 'Result'})
    away_matches['Points'] = away_matches['Result'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
    
    all_team_matches = pd.concat([home_matches, away_matches]).sort_values(['Team', 'Date'])
    
    # Calculate rolling sum of points for last 5 games (excluding current)
    all_team_matches['FormPoints'] = all_team_matches.groupby('Team')['Points'].transform(lambda x: x.rolling(5, closed='left').sum())
    
    return all_team_matches[['Date', 'Team', 'FormPoints']]

def engineer_features(df):
    """
    Main feature engineering pipeline
    """
    # 1. Rolling averages for stats
    cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']
    
    # We need to calculate these per team
    # First, transform the dataframe to be team-centric
    matches_list = []
    for index, row in df.iterrows():
        # Home team perspective
        matches_list.append({
            'Date': row['Date'],
            'Team': row['HomeTeam'],
            'Opponent': row['AwayTeam'],
            'IsHome': 1,
            'GoalsScored': row['FTHG'],
            'GoalsConceded': row['FTAG'],
            'Shots': row['HS'],
            'ShotsOnTarget': row['HST'],
            'Corners': row['HC'],
            'Result': row['FTR']
        })
        # Away team perspective
        matches_list.append({
            'Date': row['Date'],
            'Team': row['AwayTeam'],
            'Opponent': row['HomeTeam'],
            'IsHome': 0,
            'GoalsScored': row['FTAG'],
            'GoalsConceded': row['FTHG'],
            'Shots': row['AS'],
            'ShotsOnTarget': row['AST'],
            'Corners': row['AC'],
            'Result': row['FTR']
        })
    
    team_df = pd.DataFrame(matches_list)
    
    # Define features to roll
    features_to_roll = ['GoalsScored', 'GoalsConceded', 'Shots', 'ShotsOnTarget', 'Corners']
    new_cols = [f'Rolling{c}' for c in features_to_roll]
    
    # Calculate rolling averages for each team
    team_df_rolling = team_df.groupby('Team', group_keys=False).apply(
        lambda x: calculate_rolling_averages(x, features_to_roll, new_cols)
    )
    
    # Calculate Form Points
    form_df = calculate_team_form(df)
    team_df_rolling = team_df_rolling.merge(form_df, on=['Date', 'Team'], how='left')
    
    # Merge back to the original match format
    # We need to get stats for both Home and Away teams
    home_stats = team_df_rolling[team_df_rolling['IsHome'] == 1].drop(columns=['IsHome', 'Opponent', 'Result', 'GoalsScored', 'GoalsConceded', 'Shots', 'ShotsOnTarget', 'Corners'])
    away_stats = team_df_rolling[team_df_rolling['IsHome'] == 0].drop(columns=['IsHome', 'Opponent', 'Result', 'GoalsScored', 'GoalsConceded', 'Shots', 'ShotsOnTarget', 'Corners'])
    
    # Rename columns to distinguish between Home and Away
    home_stats.columns = [f'Home_{c}' if c not in ['Date', 'Team'] else ('HomeTeam' if c == 'Team' else c) for c in home_stats.columns]
    away_stats.columns = [f'Away_{c}' if c not in ['Date', 'Team'] else ('AwayTeam' if c == 'Team' else c) for c in away_stats.columns]
    
    # Merge everything back to the main df
    final_df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].merge(home_stats, on=['Date', 'HomeTeam'])
    final_df = final_df.merge(away_stats, on=['Date', 'AwayTeam'])
    
    # Target Encoding
    final_df['Target'] = final_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    
    return final_df

def main():
    try:
        df = pd.read_csv('data/matches.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        final_df = engineer_features(df)
        
        output_path = 'data/engineered_matches.csv'
        final_df.to_csv(output_path, index=False)
        print(f"Engineered {len(final_df)} matches and saved to {output_path}")
    except FileNotFoundError:
        print("data/matches.csv not found. Run data_preprocessing.py first.")
    except Exception as e:
        print(f"Error during feature engineering: {e}")

if __name__ == "__main__":
    main()
