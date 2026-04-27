import pandas as pd
import requests
import os
import io

def download_football_data(seasons, league_code='E0'):
    """
    Downloads football data from football-data.co.uk
    seasons: list of strings like ['2324', '2223', '2122']
    league_code: 'E0' for Premier League
    """
    base_url = "https://www.football-data.co.uk/mmz4281/"
    all_data = []

    for season in seasons:
        url = f"{base_url}{season}/{league_code}.csv"
        print(f"Downloading data for season {season} from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df['Season'] = season
            all_data.append(df)
            print(f"Successfully downloaded {len(df)} matches for season {season}")
        except Exception as e:
            print(f"Error downloading season {season}: {e}")

    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def clean_data(df):
    """
    Basic cleaning of the raw data
    """
    if df is None:
        return None
    
    # Select essential columns (adjust as needed)
    columns_to_keep = [
        'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
        'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
        'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'Season'
    ]
    
    # Filter columns that exist
    available_cols = [c for c in columns_to_keep if c in df.columns]
    df = df[available_cols].copy()
    
    # Convert Date to datetime
    # The format can be dd/mm/yy or dd/mm/yyyy
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    except:
        # Fallback for weird formats
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Drop rows with missing crucial data
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])
    
    return df

def main():
    seasons = ['2324', '2223', '2122', '2021', '1920', '1819']
    raw_df = download_football_data(seasons)
    
    if raw_df is not None:
        cleaned_df = clean_data(raw_df)
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        output_path = 'data/matches.csv'
        cleaned_df.to_csv(output_path, index=False)
        print(f"Saved {len(cleaned_df)} matches to {output_path}")
    else:
        print("Failed to download any data.")

if __name__ == "__main__":
    main()
