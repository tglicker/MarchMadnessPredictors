import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, team_stats_df, matchup_df):
        """
        Initialize the data processor with team statistics and matchup data.
        
        Parameters:
        -----------
        team_stats_df : pandas.DataFrame
            DataFrame containing team statistics
        matchup_df : pandas.DataFrame
            DataFrame containing matchup data with odds
        """
        self.team_stats_df = team_stats_df.copy()
        self.matchup_df = matchup_df.copy()
        self.processed_data = None
    
    def clean_team_stats(self):
        """Clean and prepare team statistics data"""
        # Make a copy to avoid modifying the original
        df = self.team_stats_df.copy()
        
        # Ensure team name column exists and is properly formatted
        if 'Team' not in df.columns and 'Team Name' not in df.columns:
            # Try to infer team name column based on string-type columns
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() > 10:
                    df.rename(columns={col: 'Team'}, inplace=True)
                    break
        
        # Rename 'Team Name' to 'Team' if it exists
        if 'Team Name' in df.columns and 'Team' not in df.columns:
            df.rename(columns={'Team Name': 'Team'}, inplace=True)
        
        # Ensure team name column exists
        if 'Team' not in df.columns:
            raise ValueError("Could not identify team name column in team statistics data")
        
        # Handle wins-losses column
        if 'Wins-Losses' in df.columns:
            # Extract wins and losses as separate columns
            wins_losses = df['Wins-Losses'].str.split('-', expand=True)
            df['Wins'] = pd.to_numeric(wins_losses[0], errors='coerce')
            df['Losses'] = pd.to_numeric(wins_losses[1], errors='coerce')
            df['Win_Rate'] = df['Wins'] / (df['Wins'] + df['Losses'])
            
            # Drop original column
            df.drop('Wins-Losses', axis=1, inplace=True)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Convert team names to consistent format
        df['Team'] = df['Team'].str.strip().str.upper()
        
        # Drop any duplicate teams
        df.drop_duplicates(subset=['Team'], inplace=True)
        
        # Set team as index for easier lookup
        df.set_index('Team', inplace=True)
        
        return df
    
    def clean_matchup_data(self):
        """Clean and prepare matchup data"""
        # Make a copy to avoid modifying the original
        df = self.matchup_df.copy()
        
        # Identify team columns
        team_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' and 'team' in col.lower():
                team_cols.append(col)
        
        # If team columns not found, try to infer
        if len(team_cols) < 2:
            string_cols = df.select_dtypes(include=['object']).columns
            if len(string_cols) >= 2:
                team_cols = list(string_cols[:2])
        
        # Rename team columns for consistency
        if len(team_cols) >= 2:
            rename_dict = {team_cols[0]: 'Team1', team_cols[1]: 'Team2'}
            df.rename(columns=rename_dict, inplace=True)
        else:
            raise ValueError("Could not identify team columns in matchup data")
        
        # Convert team names to consistent format
        df['Team1'] = df['Team1'].str.strip().str.upper()
        df['Team2'] = df['Team2'].str.strip().str.upper()
        
        # Identify betting odds columns
        spread_col = None
        ou_col = None
        ml_team1_col = None
        ml_team2_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'spread' in col_lower:
                spread_col = col
            elif any(term in col_lower for term in ['over/under', 'ou', 'o/u', 'total']):
                ou_col = col
            elif any(term in col_lower for term in ['moneyline', 'money line', 'ml']):
                if ml_team1_col is None:
                    ml_team1_col = col
                else:
                    ml_team2_col = col
        
        # Rename columns for consistency
        rename_dict = {}
        if spread_col:
            rename_dict[spread_col] = 'Spread'
        if ou_col:
            rename_dict[ou_col] = 'OverUnder'
        if ml_team1_col:
            rename_dict[ml_team1_col] = 'ML_Team1'
        if ml_team2_col:
            rename_dict[ml_team2_col] = 'ML_Team2'
        
        df.rename(columns=rename_dict, inplace=True)
        
        # Ensure necessary columns exist
        required_cols = ['Team1', 'Team2']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle numeric columns - convert to float when possible
        for col in ['Spread', 'OverUnder', 'ML_Team1', 'ML_Team2']:
            if col in df.columns:
                # Remove any non-numeric characters (like +/-)
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('+', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def merge_data(self, team_stats_df, matchup_df):
        """
        Merge team statistics with matchup data to create features for each matchup.
        
        Returns:
        --------
        pandas.DataFrame: Merged dataset with features for each team in each matchup
        """
        merged_data = []
        
        # For each matchup, get stats for both teams
        for _, row in matchup_df.iterrows():
            team1 = row['Team1']
            team2 = row['Team2']
            
            # Skip if teams not in stats data
            if team1 not in team_stats_df.index or team2 not in team_stats_df.index:
                continue
            
            # Get team stats
            team1_stats = team_stats_df.loc[team1].copy()
            team2_stats = team_stats_df.loc[team2].copy()
            
            # Create matchup features
            matchup_features = {
                'Team1': team1,
                'Team2': team2
            }
            
            # Add odds data if available
            for col in ['Spread', 'OverUnder', 'ML_Team1', 'ML_Team2']:
                if col in matchup_df.columns:
                    matchup_features[col] = row[col]
            
            # Add team stats with prefixes to differentiate
            for col in team_stats_df.columns:
                matchup_features[f'Team1_{col}'] = team1_stats[col]
                matchup_features[f'Team2_{col}'] = team2_stats[col]
            
            # Add differential features
            for col in team_stats_df.columns:
                matchup_features[f'Diff_{col}'] = team1_stats[col] - team2_stats[col]
            
            merged_data.append(matchup_features)
        
        # Convert to dataframe
        merged_df = pd.DataFrame(merged_data)
        
        return merged_df
    
    def create_target_variables(self, merged_df):
        """
        Create target variables for the four prediction tasks:
        1. Spread: whether Team1 covers the spread (1) or not (0)
        2. Money Line: whether Team1 wins outright (1) or not (0)
        3. Over/Under: whether the total score is over (1) or under (0)
        4. First to 15: whether Team1 reaches 15 points first (1) or not (0)
        
        Returns:
        --------
        tuple: (merged_df with target variables, [y_spread, y_moneyline, y_ou, y_first_to_15])
        """
        df = merged_df.copy()
        
        # Create target for spread prediction
        # Positive spread means Team1 is underdog by that many points
        if 'Spread' in df.columns:
            # For simplicity, we'll assume spread is from Team1's perspective (positive = Team1 is underdog)
            # Target is 1 if Team1 covers the spread, 0 otherwise
            # Since we don't have actual game results, we'll create a proxy based on team statistics
            
            # Create a proxy for actual point differential based on team efficiency differences
            if 'Diff_Adjusted efficiency margin' in df.columns:
                df['Estimated_Point_Diff'] = df['Diff_Adjusted efficiency margin'] * 2
            elif 'Diff_Offensive Efficiency' in df.columns and 'Diff_Defensive Efficiency' in df.columns:
                df['Estimated_Point_Diff'] = (df['Diff_Offensive Efficiency'] - df['Diff_Defensive Efficiency']) / 10
            else:
                # If neither exists, use win rate difference as a proxy
                if 'Diff_Win_Rate' in df.columns:
                    df['Estimated_Point_Diff'] = df['Diff_Win_Rate'] * 20
                else:
                    # Simply use a random variable as placeholder
                    df['Estimated_Point_Diff'] = np.random.normal(0, 5, len(df))
            
            # Team1 covers if their estimated point diff exceeds the spread
            df['Spread_Result'] = (df['Estimated_Point_Diff'] > df['Spread']).astype(int)
        else:
            # If no spread available, use a simple model based on team strength
            if 'Diff_Adjusted efficiency margin' in df.columns:
                df['Spread_Result'] = (df['Diff_Adjusted efficiency margin'] > 0).astype(int)
            elif 'Diff_Win_Rate' in df.columns:
                df['Spread_Result'] = (df['Diff_Win_Rate'] > 0).astype(int)
            else:
                # Create a random target variable as placeholder
                df['Spread_Result'] = np.random.binomial(1, 0.5, len(df))
        
        # Create target for money line prediction
        # Target is 1 if Team1 wins, 0 if Team2 wins
        if 'Diff_Adjusted efficiency margin' in df.columns:
            df['ML_Result'] = (df['Diff_Adjusted efficiency margin'] > 0).astype(int)
        elif 'Diff_Win_Rate' in df.columns:
            df['ML_Result'] = (df['Diff_Win_Rate'] > 0).astype(int)
        else:
            # Create a random target variable as placeholder
            df['ML_Result'] = np.random.binomial(1, 0.5, len(df))
        
        # Create target for over/under prediction
        # Target is 1 if over, 0 if under
        if 'OverUnder' in df.columns:
            # Use offensive and defensive efficiencies to predict total points
            if 'Team1_Offensive Efficiency' in df.columns and 'Team2_Offensive Efficiency' in df.columns:
                df['Estimated_Total'] = (df['Team1_Offensive Efficiency'] + df['Team2_Offensive Efficiency']) / 2
                df['OU_Result'] = (df['Estimated_Total'] > df['OverUnder']).astype(int)
            elif 'Team1_Adjusted Tempo' in df.columns and 'Team2_Adjusted Tempo' in df.columns:
                # Higher tempo usually means more points
                avg_tempo = (df['Team1_Adjusted Tempo'] + df['Team2_Adjusted Tempo']) / 2
                df['OU_Result'] = (avg_tempo > df['OverUnder'] / 2).astype(int)
            else:
                # Create a random target variable as placeholder
                df['OU_Result'] = np.random.binomial(1, 0.5, len(df))
        else:
            # If no OverUnder available, simply predict based on team tempos
            if 'Team1_Adjusted Tempo' in df.columns and 'Team2_Adjusted Tempo' in df.columns:
                avg_tempo = (df['Team1_Adjusted Tempo'] + df['Team2_Adjusted Tempo']) / 2
                median_tempo = avg_tempo.median()
                df['OU_Result'] = (avg_tempo > median_tempo).astype(int)
            else:
                # Create a random target variable as placeholder
                df['OU_Result'] = np.random.binomial(1, 0.5, len(df))
        
        # Create target for "First to 15 points" prediction
        # Target is 1 if Team1 reaches 15 points first, 0 if Team2 does
        if 'First_to_15_Team1' in df.columns and 'First_to_15_Team2' in df.columns:
            # Use the odds if available
            df['First_to_15_Result'] = (df['First_to_15_Team1'] < df['First_to_15_Team2']).astype(int)
        else:
            # Use a combination of tempo and offensive efficiency to predict
            # Higher tempo and offensive efficiency early in games indicates faster scoring
            has_first_half_points = 'Team1_First Half Points Per Game' in df.columns and 'Team2_First Half Points Per Game' in df.columns
            has_first_five_ppg = 'Team1_First Five Minutes PPG' in df.columns and 'Team2_First Five Minutes PPG' in df.columns
            
            if has_first_five_ppg:
                # Best predictor - first few minutes scoring rate
                df['First_to_15_Score'] = df['Team1_First Five Minutes PPG'] - df['Team2_First Five Minutes PPG']
            elif has_first_half_points:
                # Still a good predictor - first half scoring
                df['First_to_15_Score'] = df['Team1_First Half Points Per Game'] - df['Team2_First Half Points Per Game']
            elif 'Team1_Offensive Efficiency' in df.columns and 'Team2_Offensive Efficiency' in df.columns and 'Team1_Adjusted Tempo' in df.columns and 'Team2_Adjusted Tempo' in df.columns:
                # Create a composite score based on tempo and offensive efficiency
                df['Team1_Early_Score_Factor'] = df['Team1_Offensive Efficiency'] * df['Team1_Adjusted Tempo'] / 100
                df['Team2_Early_Score_Factor'] = df['Team2_Offensive Efficiency'] * df['Team2_Adjusted Tempo'] / 100
                df['First_to_15_Score'] = df['Team1_Early_Score_Factor'] - df['Team2_Early_Score_Factor']
            else:
                # Fallback to general team strength
                if 'Diff_Adjusted efficiency margin' in df.columns:
                    df['First_to_15_Score'] = df['Diff_Adjusted efficiency margin']
                else:
                    df['First_to_15_Score'] = np.random.normal(0, 1, len(df))
            
            df['First_to_15_Result'] = (df['First_to_15_Score'] > 0).astype(int)
        
        # Create target variables
        y_spread = df['Spread_Result']
        y_moneyline = df['ML_Result']
        y_ou = df['OU_Result']
        y_first_to_15 = df['First_to_15_Result']
        
        return df, [y_spread, y_moneyline, y_ou, y_first_to_15]
    
    def prepare_features(self, merged_df):
        """
        Prepare features for modeling by scaling and selecting relevant columns.
        
        Returns:
        --------
        tuple: (X_scaled, feature_names)
        """
        df = merged_df.copy()
        
        # Remove non-feature columns
        non_features = ['Team1', 'Team2', 'Spread', 'OverUnder', 'ML_Team1', 'ML_Team2', 
                        'First_to_15_Team1', 'First_to_15_Team2',
                        'Spread_Result', 'ML_Result', 'OU_Result', 'First_to_15_Result', 
                        'Estimated_Point_Diff', 'Estimated_Total', 'First_to_15_Score',
                        'Team1_Early_Score_Factor', 'Team2_Early_Score_Factor']
        
        feature_columns = [col for col in df.columns if col not in non_features]
        
        # Get features
        X = df[feature_columns]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, feature_columns
    
    def process(self):
        """
        Process the data and return the processed dataset ready for modeling.
        
        Returns:
        --------
        dict: Dictionary containing processed data components
        """
        # Clean data
        cleaned_team_stats = self.clean_team_stats()
        cleaned_matchups = self.clean_matchup_data()
        
        # Merge data
        merged_df = self.merge_data(cleaned_team_stats, cleaned_matchups)
        
        # Create target variables
        merged_df, target_variables = self.create_target_variables(merged_df)
        
        # Prepare features
        X_scaled, feature_names = self.prepare_features(merged_df)
        
        # Split data into train and test sets
        X_train, X_test, y_train_spread, y_test_spread = train_test_split(
            X_scaled, target_variables[0], test_size=0.2, random_state=42
        )
        
        _, _, y_train_ml, y_test_ml = train_test_split(
            X_scaled, target_variables[1], test_size=0.2, random_state=42
        )
        
        _, _, y_train_ou, y_test_ou = train_test_split(
            X_scaled, target_variables[2], test_size=0.2, random_state=42
        )
        
        _, _, y_train_first_to_15, y_test_first_to_15 = train_test_split(
            X_scaled, target_variables[3], test_size=0.2, random_state=42
        )
        
        # Combine y_train and y_test values
        y_train = [y_train_spread, y_train_ml, y_train_ou, y_train_first_to_15]
        y_test = [y_test_spread, y_test_ml, y_test_ou, y_test_first_to_15]
        
        # Store processed data
        self.processed_data = {
            'merged_df': merged_df,
            'X_scaled': X_scaled,
            'feature_names': feature_names,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'team_stats': cleaned_team_stats,
            'matchups': cleaned_matchups
        }
        
        return self.processed_data
