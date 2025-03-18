import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer
from predictor import Predictor

st.set_page_config(
    page_title="March Madness Betting Predictor",
    page_icon="🏀",
    layout="wide"
)

# Cache the data processing to improve performance
@st.cache_data
def process_data(team_stats_df, matchup_df):
    """Process the data and return a dictionary with processed components."""
    processor = DataProcessor(team_stats_df, matchup_df)
    return processor.process()

@st.cache_resource
def train_models(processed_data):
    """Train models on the processed data and return the trained models."""
    trainer = ModelTrainer(processed_data)
    return trainer.train_all_models()

def main():
    st.title("🏀 March Madness Betting Predictor")
    
    # Initialize session state for data persistence
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    
    # Create tabs for data input
    data_tab1, data_tab2, data_tab3 = st.tabs(["Upload Data", "Use Sample Data", "Update Matchups"])
    
    with data_tab1:
        st.header("Upload Team Statistics and Matchup Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Team Statistics File")
            team_stats_file = st.file_uploader("Upload team statistics CSV", type=["csv"])
            
        with col2:
            st.subheader("Matchup Data File")
            matchup_file = st.file_uploader("Upload matchup data CSV", type=["csv"])
        
        if team_stats_file is not None and matchup_file is not None:
            if st.button("Process Data and Train Models"):
                with st.spinner("Processing data and training models..."):
                    try:
                        # Load data
                        team_stats_df = pd.read_csv(team_stats_file)
                        matchup_df = pd.read_csv(matchup_file)
                        
                        # Process data
                        processed_data = process_data(team_stats_df, matchup_df)
                        st.session_state.processed_data = processed_data
                        
                        # Train models
                        models, X_train, X_test, y_train, y_test = train_models(processed_data)
                        st.session_state.models = models
                        
                        # Add these to processed_data for convenience
                        processed_data['X_train'] = X_train
                        processed_data['X_test'] = X_test
                        processed_data['y_train'] = y_train
                        processed_data['y_test'] = y_test
                        
                        # Create visualizer and predictor
                        visualizer = Visualizer(processed_data, models)
                        predictor = Predictor(processed_data, models)
                        
                        st.session_state.visualizer = visualizer
                        st.session_state.predictor = predictor
                        
                        st.success("Data processed and models trained successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing data: {e}")
    
    with data_tab2:
        st.header("Use Sample Data")
        st.info("Use pre-loaded sample data to get started quickly without uploading files.")
        
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data and training models..."):
                try:
                    # Load sample data
                    team_stats_df = pd.read_csv("sample_data/team_stats.csv")
                    matchup_df = pd.read_csv("sample_data/matchups.csv")
                    
                    # Process data
                    processed_data = process_data(team_stats_df, matchup_df)
                    st.session_state.processed_data = processed_data
                    
                    # Train models
                    models, X_train, X_test, y_train, y_test = train_models(processed_data)
                    st.session_state.models = models
                    
                    # Add these to processed_data for convenience
                    processed_data['X_train'] = X_train
                    processed_data['X_test'] = X_test
                    processed_data['y_train'] = y_train
                    processed_data['y_test'] = y_test
                    
                    # Create visualizer and predictor
                    visualizer = Visualizer(processed_data, models)
                    predictor = Predictor(processed_data, models)
                    
                    st.session_state.visualizer = visualizer
                    st.session_state.predictor = predictor
                    
                    st.success("Sample data loaded and models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample data: {e}")
    
    with data_tab3:
        st.header("Update Matchup Data Only")
        st.info("Use this to update odds and matchups without reloading team statistics.")
        
        if st.session_state.processed_data is not None:
            matchup_update_file = st.file_uploader("Upload updated matchup data CSV", type=["csv"], key="matchup_update")
            
            if matchup_update_file is not None:
                if st.button("Update Matchups and Retrain"):
                    with st.spinner("Updating matchups and retraining models..."):
                        try:
                            # Get stored team stats
                            team_stats_df = st.session_state.processed_data.get('team_stats_df')
                            
                            # Load new matchup data
                            matchup_df = pd.read_csv(matchup_update_file)
                            
                            # Process data
                            processed_data = process_data(team_stats_df, matchup_df)
                            st.session_state.processed_data = processed_data
                            
                            # Train models
                            models, X_train, X_test, y_train, y_test = train_models(processed_data)
                            st.session_state.models = models
                            
                            # Add these to processed_data for convenience
                            processed_data['X_train'] = X_train
                            processed_data['X_test'] = X_test
                            processed_data['y_train'] = y_train
                            processed_data['y_test'] = y_test
                            
                            # Create visualizer and predictor
                            visualizer = Visualizer(processed_data, models)
                            predictor = Predictor(processed_data, models)
                            
                            st.session_state.visualizer = visualizer
                            st.session_state.predictor = predictor
                            
                            st.success("Matchup data updated successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing the updated matchup file: {e}")
    
    # Only show analysis sections if data is loaded
    if st.session_state.processed_data is not None:
        try:
            # Get the stored data
            processed_data = st.session_state.processed_data
            models = st.session_state.models
            visualizer = st.session_state.visualizer
            predictor = st.session_state.predictor
            
            # Extract test data
            X_test = processed_data['X_test']
            y_test = processed_data['y_test']
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "Feature Importance", "Predictions", "Advanced Analysis"])
            
            with tab1:
                st.header("Model Performance")
                
                model_type = st.selectbox(
                    "Select model type to evaluate:",
                    ["Spread", "Money Line", "Over/Under", "First to 15 Points"]
                )
                
                # Evaluate model
                if model_type == "Spread":
                    model_idx = 0
                elif model_type == "Money Line":
                    model_idx = 1
                elif model_type == "Over/Under":
                    model_idx = 2
                else:  # First to 15 Points
                    model_idx = 3
                
                model = models[model_idx]
                
                # Get the appropriate y_test values
                y_model_test = y_test[model_idx]
                
                # Check if there's only one class in the test data
                unique_test_classes = np.unique(y_model_test)
                
                if len(unique_test_classes) < 2:
                    # Only one class in test data, show special message
                    st.warning(f"The {model_type} dataset contains only one class ({unique_test_classes[0]}). " +
                              "This means all samples belong to the same category. The model will always predict this class.")
                    
                    # Set metrics for the one-class scenario
                    accuracy = 1.0 if np.all(y_model_test == model.predict(X_test)) else 0.0
                    precision = 1.0 if unique_test_classes[0] == 1 else 0.0  # If all 0s, precision is 0
                    recall = 1.0 if unique_test_classes[0] == 1 else 0.0     # If all 0s, recall is 0
                    f1 = 1.0 if unique_test_classes[0] == 1 else 0.0         # If all 0s, F1 is 0
                    auc = 0.5  # Random AUC
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = np.zeros((len(y_model_test), 2))
                    y_pred_proba[:, unique_test_classes[0]] = 1.0
                    y_pred_proba = y_pred_proba[:, 1]  # Get second column for consistency
                else:
                    # Normal case with multiple classes
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_model_test, y_pred)
                    precision = precision_score(y_model_test, y_pred, zero_division=0)
                    recall = recall_score(y_model_test, y_pred, zero_division=0)
                    f1 = f1_score(y_model_test, y_pred, zero_division=0)
                    try:
                        auc = roc_auc_score(y_model_test, y_pred_proba)
                    except:
                        auc = 0.5  # Default to random classifier AUC
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{accuracy:.3f}")
                col2.metric("Precision", f"{precision:.3f}")
                col3.metric("Recall", f"{recall:.3f}")
                col4.metric("F1 Score", f"{f1:.3f}")
                col5.metric("AUC-ROC", f"{auc:.3f}")
                
                # Confusion matrix
                cm = confusion_matrix(y_model_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.matshow(cm, cmap=plt.cm.Blues)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), va='center', ha='center')
                ax.set_title('Confusion Matrix')
                if model_type == "Spread":
                    ax.set_xlabel('Predicted (0=Away, 1=Home)')
                    ax.set_ylabel('Actual (0=Away, 1=Home)')
                elif model_type == "Money Line":
                    ax.set_xlabel('Predicted (0=Team1 loss, 1=Team1 win)')
                    ax.set_ylabel('Actual (0=Team1 loss, 1=Team1 win)')
                elif model_type == "Over/Under":
                    ax.set_xlabel('Predicted (0=Under, 1=Over)')
                    ax.set_ylabel('Actual (0=Under, 1=Over)')
                else:  # First to 15 Points
                    ax.set_xlabel('Predicted (0=Team2 first, 1=Team1 first)')
                    ax.set_ylabel('Actual (0=Team2 first, 1=Team1 first)')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                st.pyplot(fig)
                
                # Cross-validation results
                st.subheader("Cross-Validation Performance")
                cv_results = visualizer.plot_cross_validation_results(model_idx)
                st.plotly_chart(cv_results)
            
            with tab2:
                st.header("Feature Importance")
                
                model_type = st.selectbox(
                    "Select model:",
                    ["Spread", "Money Line", "Over/Under", "First to 15 Points"],
                    key="feature_importance_model"
                )
                
                if model_type == "Spread":
                    model_idx = 0
                elif model_type == "Money Line":
                    model_idx = 1
                elif model_type == "Over/Under":
                    model_idx = 2
                else:  # First to 15 Points
                    model_idx = 3
                
                # Get feature importance for the selected model
                feature_imp = visualizer.get_feature_importance(model_idx) 
                
                # Display current feature importance visualization
                st.subheader("Current Feature Importance")
                feature_importance_fig = visualizer.plot_feature_importance(model_idx)
                st.plotly_chart(feature_importance_fig)
                
                # Add interactive feature importance adjustment
                st.subheader("Adjust Feature Importance")
                st.info("Use the sliders below to adjust the importance of each feature in the model. " +
                       "This won't change the trained model but will help you explore 'what-if' scenarios " +
                       "by giving you control over how much each factor influences the predictions.")
                
                # Initialize feature weights in session state if not present
                if 'feature_weights' not in st.session_state or st.session_state.feature_weights is None:
                    # Get the feature names from the importance dict
                    feature_names = []
                    feature_values = []
                    
                    for feature, importance in feature_imp.items():
                        feature_names.append(feature)
                        feature_values.append(1.0)  # Start all weights at 1.0 (no adjustment)
                    
                    st.session_state.feature_weights = {
                        'feature_names': feature_names,
                        'weights': {name: 1.0 for name in feature_names},
                        'model_idx': model_idx
                    }
                
                # Check if model changed and reset weights if needed
                if st.session_state.feature_weights.get('model_idx') != model_idx:
                    # Get the feature names from the importance dict
                    feature_names = []
                    
                    for feature, importance in feature_imp.items():
                        feature_names.append(feature)
                    
                    st.session_state.feature_weights = {
                        'feature_names': feature_names,
                        'weights': {name: 1.0 for name in feature_names},
                        'model_idx': model_idx
                    }
                
                # Create sliders for each feature
                weights_changed = False
                
                # Show the top features only to avoid clutter
                top_features = sorted(feature_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                
                for feature, importance in top_features:
                    current_weight = st.session_state.feature_weights['weights'].get(feature, 1.0)
                    new_weight = st.slider(
                        f"{feature} (base importance: {importance:.4f})",
                        min_value=0.0,
                        max_value=2.0,
                        value=current_weight,
                        step=0.1,
                        key=f"slider_{feature}_{model_idx}"
                    )
                    
                    if new_weight != current_weight:
                        st.session_state.feature_weights['weights'][feature] = new_weight
                        weights_changed = True
                
                # Apply weights button
                if st.button("Apply Weight Adjustments", key=f"apply_weights_{model_idx}"):
                    st.session_state.weight_adjustments_applied = True
                    
                    # Create an adjusted feature importance chart
                    adjusted_importance = {}
                    for feature, importance in feature_imp.items():
                        weight = st.session_state.feature_weights['weights'].get(feature, 1.0)
                        adjusted_importance[feature] = importance * weight
                    
                    # Sort the dict by values
                    adjusted_importance = dict(sorted(adjusted_importance.items(), key=lambda x: abs(x[1]), reverse=True))
                    
                    # Create a side-by-side comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Importance")
                        original_fig = go.Figure()
                        original_fig.add_trace(go.Bar(
                            x=list(feature_imp.keys()),
                            y=list(feature_imp.values()),
                            marker_color=['red' if x < 0 else 'blue' for x in feature_imp.values()]
                        ))
                        original_fig.update_layout(
                            title="Original Feature Importance",
                            xaxis=dict(title="Features"),
                            yaxis=dict(title="Importance"),
                            height=500
                        )
                        st.plotly_chart(original_fig)
                    
                    with col2:
                        st.subheader("Adjusted Importance")
                        adjusted_fig = go.Figure()
                        adjusted_fig.add_trace(go.Bar(
                            x=list(adjusted_importance.keys()),
                            y=list(adjusted_importance.values()),
                            marker_color=['red' if x < 0 else 'blue' for x in adjusted_importance.values()]
                        ))
                        adjusted_fig.update_layout(
                            title="Adjusted Feature Importance",
                            xaxis=dict(title="Features"),
                            yaxis=dict(title="Adjusted Importance"),
                            height=500
                        )
                        st.plotly_chart(adjusted_fig)
                    
                    st.session_state.adjusted_importance = adjusted_importance
                
                # Feature correlation matrix
                st.subheader("Feature Correlation Matrix")
                corr_fig = visualizer.plot_correlation_matrix()
                st.plotly_chart(corr_fig)
            
            with tab3:
                st.header("Predictions for Upcoming Matchups")
                
                # Get all matchups
                all_matchups = predictor.get_all_matchups()
                
                # Remove duplicate matchups (e.g., "Team A vs Team B" and "Team B vs Team A")
                # Create a unique set of team pairs
                unique_matchups = []
                seen_pairs = set()
                
                for matchup in all_matchups:
                    teams = matchup.split(" vs ")
                    if len(teams) == 2:
                        # Create a frozenset to identify unique pairs regardless of order
                        team_pair = frozenset([teams[0].upper(), teams[1].upper()])
                        
                        if team_pair not in seen_pairs:
                            seen_pairs.add(team_pair)
                            unique_matchups.append(matchup)
                
                # Add search functionality
                st.markdown("### Find a Specific Matchup")
                search_term = st.text_input("Search for a team (e.g., 'Texas Tech', 'UNC')", key='team_search')
                
                # Filter matchups based on search
                if search_term:
                    search_term = search_term.upper()
                    filtered_matchups = [m for m in unique_matchups if search_term in m.upper()]
                    
                    # Show number of matches
                    if len(filtered_matchups) > 0:
                        st.success(f"Found {len(filtered_matchups)} matchups containing '{search_term}'")
                    else:
                        st.warning(f"No matchups found containing '{search_term}'")
                        # Show suggestions for partial matches
                        partial_matches = [m for m in unique_matchups if any(term in m.upper() 
                                                                          for term in search_term.split())]
                        if partial_matches:
                            st.info("Showing partial matches instead:")
                            filtered_matchups = partial_matches
                        else:
                            filtered_matchups = unique_matchups  # Fall back to all matchups
                else:
                    filtered_matchups = unique_matchups
                
                # Display a note about the searchable dropdown
                st.info("💡 TIP: You can also type directly in the dropdown below to search for teams")
                
                # Select matchup from filtered list
                selected_matchup = st.selectbox(
                    "Select matchup to predict:",
                    filtered_matchups
                )
                
                # Make predictions for the selected matchup
                predictions = predictor.predict_matchup(selected_matchup)
                
                # Display predictions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Spread Prediction")
                    st.markdown(f"**Current Spread:** {predictions['spread_value']}")
                    team_to_bet = predictions['team1'] if predictions['spread_prediction'] == 1 else predictions['team2']
                    st.markdown(f"**Team to bet on:** {team_to_bet}")
                    st.markdown(f"**Confidence:** {predictions['spread_confidence']:.2%}")
                    
                    # Calculate potential winnings on a $25 bet (typically -110 odds for spread)
                    odds = -110  # Standard spread odds
                    if odds < 0:
                        potential_profit = (25 * 100) / abs(odds)
                    else:
                        potential_profit = (25 * odds) / 100
                    st.markdown(f"**Potential Winnings on $25 bet:** ${potential_profit:.2f}")
                
                with col2:
                    st.subheader("Money Line Prediction")
                    ml_team1 = predictions['money_line_team1']
                    ml_team2 = predictions['money_line_team2']
                    st.markdown(f"**{predictions['team1']} ML:** {ml_team1}")
                    st.markdown(f"**{predictions['team2']} ML:** {ml_team2}")
                    
                    winner = predictions['team1'] if predictions['money_line_prediction'] == 1 else predictions['team2']
                    winner_odds = ml_team1 if predictions['money_line_prediction'] == 1 else ml_team2
                    st.markdown(f"**Predicted Winner:** {winner}")
                    st.markdown(f"**Confidence:** {predictions['money_line_confidence']:.2%}")
                    
                    # Calculate potential winnings on a $25 bet
                    try:
                        if isinstance(winner_odds, (int, float)):
                            if winner_odds > 0:
                                potential_profit = (25 * winner_odds) / 100
                            else:
                                potential_profit = (25 * 100) / abs(winner_odds)
                            
                            st.markdown(f"**Potential Winnings on $25 bet:** ${potential_profit:.2f}")
                        else:
                            st.markdown("**Potential Winnings:** Unable to calculate (invalid odds format)")
                    except Exception as e:
                        st.markdown("**Potential Winnings:** Unable to calculate")
                
                with col3:
                    st.subheader("Over/Under Prediction")
                    st.markdown(f"**Current Over/Under:** {predictions['over_under_value']}")
                    bet_type = "Over" if predictions['over_under_prediction'] == 1 else "Under"
                    st.markdown(f"**Prediction:** {bet_type}")
                    st.markdown(f"**Confidence:** {predictions['over_under_confidence']:.2%}")
                    
                    # Calculate potential winnings on a $25 bet (typically -110 odds for over/under)
                    odds = -110  # Standard over/under odds
                    if odds < 0:
                        potential_profit = (25 * 100) / abs(odds)
                    else:
                        potential_profit = (25 * odds) / 100
                    st.markdown(f"**Potential Winnings on $25 bet:** ${potential_profit:.2f}")
                
                # Display First to 15 Points prediction in a separate section
                st.subheader("First to 15 Points Prediction")
                
                # Check if first_to_15 prediction is available in the predictions
                if 'first_to_15_prediction' in predictions:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        team_first = predictions['team1'] if predictions['first_to_15_prediction'] == 1 else predictions['team2']
                        st.markdown(f"**Team predicted to reach 15 first:** {team_first}")
                        st.markdown(f"**Confidence:** {predictions.get('first_to_15_confidence', 0.0):.2%}")
                    
                    with col2:
                        # Calculate potential winnings on a $25 bet (typically -110 odds for first to 15)
                        odds = -110  # Standard prop bet odds
                        if odds < 0:
                            potential_profit = (25 * 100) / abs(odds)
                        else:
                            potential_profit = (25 * odds) / 100
                        st.markdown(f"**Potential Winnings on $25 bet:** ${potential_profit:.2f}")
                        
                        # Display key contributing factors if available
                        st.markdown("**Key factors:** Tempo, Offensive Efficiency, 3pt Shooting %")
                else:
                    st.info("First to 15 Points prediction not available for this matchup.")
                
                # Display matchup details
                st.subheader("Matchup Details")
                matchup_details = predictor.get_matchup_details(selected_matchup)
                
                # Create a comparison table of key statistics
                comparison_df = pd.DataFrame()
                for key, value in matchup_details.items():
                    if not key.startswith('team') and not key.startswith('_'):
                        # Skip metadata, focus on stats
                        if isinstance(value, dict):
                            comparison_df.loc[key, matchup_details['team1']] = value.get('team1', '')
                            comparison_df.loc[key, matchup_details['team2']] = value.get('team2', '')
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Radar chart comparison
                st.subheader("Team Comparison")
                radar_chart = visualizer.plot_team_comparison_radar(selected_matchup)
                st.plotly_chart(radar_chart)
            
            with tab4:
                st.header("Advanced Analysis")
                
                # Value bets section
                st.subheader("Value Bets Identified")
                st.markdown("""
                Value bets are where our model's confidence exceeds a threshold and disagrees with the implied probability from the betting odds.
                These represent potential opportunities where the market may be mispriced.
                """)
                
                confidence_threshold = st.slider("Confidence Threshold", 0.6, 0.95, 0.75, 0.05)
                
                value_bets = predictor.find_value_bets(confidence_threshold)
                
                if len(value_bets) > 0:
                    # Create a bar chart showing value scores
                    teams = [f"{bet['matchup']}" for bet in value_bets]
                    value_scores = [bet['value_score'] for bet in value_bets]
                    bet_types = [bet['bet_type'] for bet in value_bets]
                    
                    fig = go.Figure()
                    
                    # Add bars for each bet type with different colors
                    for bet_type in set(bet_types):
                        indices = [i for i, bt in enumerate(bet_types) if bt == bet_type]
                        fig.add_trace(go.Bar(
                            x=[teams[i] for i in indices],
                            y=[value_scores[i] for i in indices],
                            name=bet_type
                        ))
                    
                    fig.update_layout(
                        title="Value Bets by Matchup",
                        xaxis_title="Matchup",
                        yaxis_title="Value Score (higher is better)",
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig)
                else:
                    st.write("No high-value bets found.")
                
                # Model performance by team
                st.subheader("Model Performance by Team")
                team_performance = visualizer.plot_team_performance()
                st.plotly_chart(team_performance)
                
                # Betting strategy simulator
                st.subheader("Simple Betting Strategy Simulator")
                st.markdown("""
                This simulator shows how the model would perform with different betting strategies.
                """)
                
                bet_amount = st.slider("Bet amount ($)", 10, 500, 100, 10)
                confidence_threshold = st.slider("Confidence threshold", 0.5, 0.95, 0.7, 0.05, key="simulation_confidence")
                
                simulation_results = predictor.simulate_betting_strategy(bet_amount, confidence_threshold)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Profit/Loss", f"${simulation_results['profit_loss']:.2f}")
                col2.metric("ROI", f"{simulation_results['roi']:.2%}")
                col3.metric("Win Rate", f"{simulation_results['win_rate']:.2%}")
                
                # Plot simulation results
                fig = px.line(
                    x=simulation_results['matchup_indices'],
                    y=simulation_results['cumulative_profit'],
                    labels={"x": "Matchup Number", "y": "Cumulative Profit/Loss ($)"},
                    title="Simulated Betting Performance"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
