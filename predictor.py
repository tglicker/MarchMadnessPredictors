import numpy as np
import pandas as pd

class Predictor:
    def __init__(self, processed_data, models):
        """
        Initialize the predictor with processed data and trained models.
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary containing processed data components
        models : list
            List of trained models [spread_model, moneyline_model, over_under_model]
        """
        self.processed_data = processed_data
        self.models = models
        self.merged_df = processed_data['merged_df']
        self.feature_names = processed_data['feature_names']
    
    def get_all_matchups(self):
        """
        Get a list of all matchups in the format "Team1 vs Team2".
        
        Returns:
        --------
        list: List of matchup strings
        """
        # Check if merged_df exists and has the required columns
        if self.merged_df is None or 'Team1' not in self.merged_df.columns or 'Team2' not in self.merged_df.columns:
            print("Warning: Missing required team columns in data")
            return []
            
        # Create matchup strings
        matchups = []
        seen_pairs = set()  # Track unique team pairs (regardless of order)
        
        # Create a dictionary to store all team variants
        team_variations = {}
        
        # First pass: collect all team names and variations
        for _, row in self.merged_df.iterrows():
            for team_col in ['Team1', 'Team2']:
                team = row[team_col]
                if pd.isna(team) or team == '':
                    continue
                    
                # Normalize the team name for matching
                team_norm = str(team).strip().upper()
                
                # Store each variant with its normalized version
                if team_norm not in team_variations:
                    team_variations[team_norm] = set()
                team_variations[team_norm].add(str(team).strip())
        
        # Generate all matchups from the dataframe
        for _, row in self.merged_df.iterrows():
            team1 = row['Team1']
            team2 = row['Team2']
            
            # Skip empty team names
            if pd.isna(team1) or pd.isna(team2) or team1 == '' or team2 == '':
                continue
                
            # Clean and standardize team names for display
            team1 = str(team1).strip()
            team2 = str(team2).strip()
                
            # Create matchup string
            matchup = f"{team1} vs {team2}"
            
            # Add to matchups if not already included
            if matchup not in matchups:
                matchups.append(matchup)
                
            # Also consider the reverse matchup (team2 vs team1)
            # Only add if we haven't seen this pair of teams before
            team_pair = frozenset([team1.upper(), team2.upper()])
            if team_pair not in seen_pairs:
                seen_pairs.add(team_pair)
                reverse_matchup = f"{team2} vs {team1}"
                if reverse_matchup not in matchups:
                    matchups.append(reverse_matchup)
                    
        # Also add combinations with variant spellings
        for team_pair in seen_pairs:
            if len(team_pair) != 2:
                continue
                
            team_list = list(team_pair)
            team1_norm, team2_norm = team_list[0], team_list[1]
            
            # Get all variants of these teams
            team1_variants = team_variations.get(team1_norm, {team1_norm})
            team2_variants = team_variations.get(team2_norm, {team2_norm})
            
            # Create additional matchups with variant spellings
            for t1 in team1_variants:
                for t2 in team2_variants:
                    variant_matchup1 = f"{t1} vs {t2}"
                    variant_matchup2 = f"{t2} vs {t1}"
                    
                    if variant_matchup1 not in matchups:
                        matchups.append(variant_matchup1)
                    if variant_matchup2 not in matchups:
                        matchups.append(variant_matchup2)
        
        # Sort alphabetically for consistent display
        return sorted(matchups)
    
    def get_matchup_index(self, matchup):
        """
        Get the index of a matchup in the merged dataframe.
        
        Parameters:
        -----------
        matchup : str
            String in format "Team1 vs Team2"
        
        Returns:
        --------
        int: Index of the matchup in the merged dataframe
        """
        teams = matchup.split(' vs ')
        team1 = teams[0].strip()
        team2 = teams[1].strip()
        
        # First try exact match
        matchup_idx = self.merged_df[(self.merged_df['Team1'] == team1) & 
                                      (self.merged_df['Team2'] == team2)].index
        
        if len(matchup_idx) == 0:
            # Try the reverse order
            matchup_idx = self.merged_df[(self.merged_df['Team1'] == team2) & 
                                         (self.merged_df['Team2'] == team1)].index
        
        # If not found, try case-insensitive match
        if len(matchup_idx) == 0:
            print(f"Trying case-insensitive match for '{matchup}'")
            team1_upper = team1.upper()
            team2_upper = team2.upper()
            
            # Create case-insensitive team matchers
            for i, row in self.merged_df.iterrows():
                df_team1 = str(row['Team1']).strip().upper()
                df_team2 = str(row['Team2']).strip().upper()
                
                # Check both team orders
                if (df_team1 == team1_upper and df_team2 == team2_upper) or \
                   (df_team1 == team2_upper and df_team2 == team1_upper):
                    matchup_idx = [i]
                    break
        
        # If still not found, try fuzzy match based on substrings
        if len(matchup_idx) == 0:
            print(f"Trying fuzzy match for '{matchup}'")
            for i, row in self.merged_df.iterrows():
                df_team1 = str(row['Team1']).strip().upper()
                df_team2 = str(row['Team2']).strip().upper()
                
                # Check if team1 is a substring of df_team1 or df_team2
                # and team2 is a substring of the other
                team1_in_df1 = team1_upper in df_team1 or df_team1 in team1_upper
                team1_in_df2 = team1_upper in df_team2 or df_team2 in team1_upper
                team2_in_df1 = team2_upper in df_team1 or df_team1 in team2_upper
                team2_in_df2 = team2_upper in df_team2 or df_team2 in team2_upper
                
                if (team1_in_df1 and team2_in_df2) or (team1_in_df2 and team2_in_df1):
                    matchup_idx = [i]
                    break
        
        if len(matchup_idx) == 0:
            raise ValueError(f"Matchup '{matchup}' not found in the data. "
                            f"Please check team names and try again.")
        
        return matchup_idx[0]
    
    def get_matchup_features(self, matchup_idx):
        """
        Get the features for a specific matchup.
        
        Parameters:
        -----------
        matchup_idx : int
            Index of the matchup in the merged dataframe
        
        Returns:
        --------
        numpy.ndarray: Feature vector for the matchup
        """
        X_scaled = self.processed_data['X_scaled']
        
        # Find the corresponding row in X_scaled
        # This assumes X_scaled has the same order as merged_df
        if matchup_idx < len(X_scaled):
            return X_scaled[matchup_idx].reshape(1, -1)
        else:
            # If index is out of bounds, use the row from merged_df directly
            row = self.merged_df.iloc[matchup_idx]
            features = np.array([row[feat] for feat in self.feature_names]).reshape(1, -1)
            return features
    
    def predict_matchup(self, matchup):
        """
        Make predictions for a specific matchup.
        
        Parameters:
        -----------
        matchup : str
            String in format "Team1 vs Team2"
        
        Returns:
        --------
        dict: Dictionary with prediction results
        """
        # Get the matchup index
        matchup_idx = self.get_matchup_index(matchup)
        
        # Get the matchup row
        matchup_row = self.merged_df.iloc[matchup_idx]
        
        # Get the feature vector
        X = self.get_matchup_features(matchup_idx)
        
        # Make predictions
        spread_pred = self.models[0].predict(X)[0]
        spread_prob = self.models[0].predict_proba(X)[0][spread_pred]
        
        ml_pred = self.models[1].predict(X)[0]
        ml_prob = self.models[1].predict_proba(X)[0][ml_pred]
        
        ou_pred = self.models[2].predict(X)[0]
        ou_prob = self.models[2].predict_proba(X)[0][ou_pred]
        
        # First to 15 Points prediction (if model exists)
        first_to_15_pred = None
        first_to_15_prob = None
        if len(self.models) > 3:  # Check if we have the First to 15 model
            first_to_15_pred = self.models[3].predict(X)[0]
            first_to_15_prob = self.models[3].predict_proba(X)[0][first_to_15_pred]
        
        # Extract teams
        team1 = matchup_row['Team1']
        team2 = matchup_row['Team2']
        
        # Extract odds
        spread_value = matchup_row.get('Spread', 'N/A')
        money_line_team1 = matchup_row.get('ML_Team1', 'N/A')
        money_line_team2 = matchup_row.get('ML_Team2', 'N/A')
        over_under_value = matchup_row.get('OverUnder', 'N/A')
        
        # Create prediction result
        result = {
            'team1': team1,
            'team2': team2,
            'spread_value': spread_value,
            'spread_prediction': int(spread_pred),
            'spread_confidence': float(spread_prob),
            'money_line_team1': money_line_team1,
            'money_line_team2': money_line_team2,
            'money_line_prediction': int(ml_pred),
            'money_line_confidence': float(ml_prob),
            'over_under_value': over_under_value,
            'over_under_prediction': int(ou_pred),
            'over_under_confidence': float(ou_prob)
        }
        
        # Add First to 15 prediction if available
        if first_to_15_pred is not None and first_to_15_prob is not None:
            result['first_to_15_prediction'] = int(first_to_15_pred)
            result['first_to_15_confidence'] = float(first_to_15_prob)
        
        return result
    
    def get_matchup_details(self, matchup):
        """
        Get detailed information about a matchup.
        
        Parameters:
        -----------
        matchup : str
            String in format "Team1 vs Team2"
        
        Returns:
        --------
        dict: Dictionary with matchup details
        """
        # Get the matchup index
        matchup_idx = self.get_matchup_index(matchup)
        
        # Get the matchup row
        matchup_row = self.merged_df.iloc[matchup_idx]
        
        # Extract teams
        team1 = matchup_row['Team1']
        team2 = matchup_row['Team2']
        
        # Collect relevant stats
        stats = {}
        
        # Find all stats for both teams
        for col in matchup_row.index:
            if col.startswith('Team1_'):
                stat_name = col.replace('Team1_', '')
                if f'Team2_{stat_name}' in matchup_row.index:
                    stats[stat_name] = {
                        'team1': matchup_row[col],
                        'team2': matchup_row[f'Team2_{stat_name}']
                    }
        
        # Add betting odds
        if 'Spread' in matchup_row:
            stats['Spread'] = {
                'team1': matchup_row['Spread'],
                'team2': -matchup_row['Spread'] if isinstance(matchup_row['Spread'], (int, float)) else matchup_row['Spread']
            }
        
        if 'ML_Team1' in matchup_row and 'ML_Team2' in matchup_row:
            stats['Money Line'] = {
                'team1': matchup_row['ML_Team1'],
                'team2': matchup_row['ML_Team2']
            }
        
        if 'OverUnder' in matchup_row:
            stats['Over/Under'] = {
                'team1': matchup_row['OverUnder'],
                'team2': matchup_row['OverUnder']
            }

#This IS ALL ADDED AND CAN BE DELETED
                # Create comparison table directly
        comparison_df = pd.DataFrame(index=stats.keys(), columns=[team1, team2])
        
        # Fill in values 
        for stat_name, values in stats.items():
            if isinstance(values, dict):
                comparison_df.loc[stat_name, team1] = values.get('team1')
                comparison_df.loc[stat_name, team2] = values.get('team2')
        
        # Clean up any NaN or None values
        comparison_df = comparison_df.fillna('')
        
        # Format numeric values
        for col in comparison_df.columns:
            for idx in comparison_df.index:
                val = comparison_df.loc[idx, col]
                if isinstance(val, (int, float)):
                    comparison_df.loc[idx, col] = f"{val:.2f}"
#This IS ALL ADDED AND CAN BE DELETED
        
        # Create matchup details
        details = {
            'team1': team1,
            'team2': team2,
            'stats': stats,
            'comparison_df': comparison_df #This IS ALL ADDED AND CAN BE DELETED
        }
        
        return details
    
    def find_value_bets(self, confidence_threshold=0.7):
        """
        Find potential value bets where model confidence is high.
        
        Parameters:
        -----------
        confidence_threshold : float
            Minimum confidence threshold for identifying value bets
        
        Returns:
        --------
        list: List of dictionaries with value bet details
        """
        value_bets = []
        
        for matchup in self.get_all_matchups():
            try:
                prediction = self.predict_matchup(matchup)
                
                # Check for value bets
                # Spread bet
                if prediction['spread_confidence'] > confidence_threshold:
                    value_bets.append({
                        'Matchup': matchup,
                        'Bet Type': 'Spread',
                        'Prediction': 'Home team covers' if prediction['spread_prediction'] == 1 else 'Away team covers',
                        'Confidence': prediction['spread_confidence'],
                        'Value Score': prediction['spread_confidence'] * 10,
                        'Odds Value': prediction.get('spread_value', 'N/A')
                    })
                
                # Money line bet
                if prediction['money_line_confidence'] > confidence_threshold:
                    # Calculate implied probability from money line odds
                    ml_value = prediction['money_line_team1'] if prediction['money_line_prediction'] == 1 else prediction['money_line_team2']
                    
                    if isinstance(ml_value, (int, float)):
                        # Convert American odds to probability
                        if ml_value > 0:
                            implied_prob = 100 / (ml_value + 100)
                        else:
                            implied_prob = -ml_value / (-ml_value + 100)
                        
                        # Calculate value score: model confidence / implied probability
                        # Higher values indicate better value
                        value_score = prediction['money_line_confidence'] / implied_prob if implied_prob > 0 else 0
                    else:
                        value_score = prediction['money_line_confidence'] * 10
                    
                    value_bets.append({
                        'Matchup': matchup,
                        'Bet Type': 'Money Line',
                        'Prediction': f"{prediction['team1']} wins" if prediction['money_line_prediction'] == 1 else f"{prediction['team2']} wins",
                        'Confidence': prediction['money_line_confidence'],
                        'Value Score': value_score,
                        'Odds Value': ml_value
                    })
                
                # Over/Under bet
                if prediction['over_under_confidence'] > confidence_threshold:
                    value_bets.append({
                        'Matchup': matchup,
                        'Bet Type': 'Over/Under',
                        'Prediction': 'OVER' if prediction['over_under_prediction'] == 1 else 'UNDER',
                        'Confidence': prediction['over_under_confidence'],
                        'Value Score': prediction['over_under_confidence'] * 10,
                        'Odds Value': prediction.get('over_under_value', 'N/A')
                    })
                    
                # First to 15 Points bet
                if 'first_to_15_prediction' in prediction and prediction.get('first_to_15_confidence', 0) > confidence_threshold:
                    team_first = prediction['team1'] if prediction['first_to_15_prediction'] == 1 else prediction['team2']
                    value_bets.append({
                        'Matchup': matchup,
                        'Bet Type': 'First to 15',
                        'Prediction': f"{team_first} scores 15 first",
                        'Confidence': prediction['first_to_15_confidence'],
                        'Value Score': prediction['first_to_15_confidence'] * 10,
                        'Odds Value': '-110'  # Default odds for prop bets
                    })
            except Exception as e:
                # Skip matchups that cause errors
                continue
        
        # Sort by value score
        value_bets.sort(key=lambda x: x['Value Score'], reverse=True)
        
        return value_bets
    
    def simulate_betting_strategy(self, bet_amount=100, confidence_threshold=0.7):
        """
        Simulate a simple betting strategy on test data.
        
        Parameters:
        -----------
        bet_amount : float
            Amount to bet on each game
        confidence_threshold : float
            Minimum confidence threshold for placing a bet
        
        Returns:
        --------
        dict: Dictionary with simulation results
        """
        # Get test data
        X_test = self.processed_data['X_test']
        y_test = self.processed_data['y_test']
        
        # Initialize variables
        total_bets = 0
        winning_bets = 0
        profit_loss = 0
        
        # Initialize list to track cumulative profit/loss
        cumulative_profit = []
        matchup_indices = []
        
        # Simulate bets for all models
        for model_idx, model in enumerate(self.models):
            # Skip if we're beyond our basic models or y_test isn't available for this model
            if model_idx >= len(y_test) or model_idx >= 4:
                continue
                
            # Get the model name
            model_names = ['Spread', 'Money Line', 'Over/Under', 'First to 15']
            model_name = model_names[model_idx] if model_idx < len(model_names) else f"Model {model_idx}"
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate payout multipliers (simplified)
            payout_multipliers = np.ones(len(y_test[model_idx])) * 1.9  # Default payout (like -110 odds)
            
            # Loop through each prediction
            for i in range(len(y_test[model_idx])):
                # Get prediction probability
                pred_class = y_pred[i]
                confidence = y_pred_proba[i][pred_class]
                
                # Check if confidence meets threshold
                if confidence >= confidence_threshold:
                    total_bets += 1
                    matchup_indices.append(total_bets)
                    
                    # Check if prediction is correct
                    if y_pred[i] == y_test[model_idx][i]:
                        winning_bets += 1
                        profit = bet_amount * (payout_multipliers[i] - 1)
                        profit_loss += profit
                    else:
                        profit_loss -= bet_amount
                    
                    # Track cumulative profit/loss
                    cumulative_profit.append(profit_loss)
        
        # Calculate results
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        roi = profit_loss / (total_bets * bet_amount) if total_bets > 0 else 0
        
        results = {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'profit_loss': profit_loss,
            'win_rate': win_rate,
            'roi': roi,
            'cumulative_profit': cumulative_profit,
            'matchup_indices': matchup_indices
        }
        
        return results
