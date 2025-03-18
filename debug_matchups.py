print('Debugging matchups in predictor.py')
import pandas as pd
import numpy as np
from predictor import Predictor
from data_processor import DataProcessor
from model_trainer import ModelTrainer

# Create sample data with Texas Tech and UNC Wilmington
team_stats = pd.DataFrame({
    'Team': ['Texas Tech', 'UNC Wilmington', 'Duke', 'Kentucky'],
    'Stat1': [10, 15, 12, 14],
    'Stat2': [5, 7, 6, 8]
})

matchup_data = pd.DataFrame({
    'Team1': ['Texas Tech', 'Duke'], 
    'Team2': ['UNC Wilmington', 'Kentucky'],
    'Spread': [5.5, 3.5],
    'ML_Team1': [150, 120],
    'ML_Team2': [-170, -130],
    'OverUnder': [140, 145]
})

# Process data (simplified)
processor = DataProcessor(team_stats, matchup_data)
processed_data = processor.process()

# Train models
trainer = ModelTrainer(processed_data)
models, _, _, _, _ = trainer.train_all_models()

# Create predictor
predictor = Predictor(processed_data, models)

# Debug: Print all matchups
print('All matchups:')
all_matchups = predictor.get_all_matchups()
for m in all_matchups:
    print(f'  - {m}')

# Debug: Print data in merged_df
print('\nMerged dataframe:')
for i, row in processed_data['merged_df'].iterrows():
    print(f'  {i}: {row["Team1"]} vs {row["Team2"]}')

