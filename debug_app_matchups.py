print('Deep debugging matchups with app.py logic')
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from predictor import Predictor

# Function to check if a string contains any of the target team names
def contains_team(s, targets):
    if not isinstance(s, str):
        return False
    s = s.upper()
    return any(target.upper() in s for target in targets)

# Get matchups from the app
class FakeCache:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# Monkey patch st.cache_data and st.cache_resource
st.cache_data = FakeCache
st.cache_resource = FakeCache

# Import app functions
from app import process_data, train_models

# Search for specific teams in the data
teams_to_find = ['Texas Tech', 'UNC Wilmington']

try:
    # We can't get the actual uploaded files here, so we'll just debug the get_all_matchups function
    print('\nChecking the predictor.get_all_matchups implementation:')
    print('1. The function should return ALL team combinations, including reversed order')
    print('2. Let\'s look at the code to see if there are any potential issues:')
    with open('predictor.py', 'r') as f:
        matchup_code = False
        for line in f:
            if 'def get_all_matchups' in line:
                matchup_code = True
            elif matchup_code and 'def ' in line:
                matchup_code = False
            if matchup_code:
                print(f'    {line.rstrip()}')
                
    print('\nPotential issues:')
    print('- Input data might have team name case mismatches: Check if "Texas Tech" vs "TEXAS TECH"')
    print('- Names might be slightly different: e.g., "UNC-Wilmington" vs "UNC Wilmington"')
    print('- Team might be missing entirely from the dataset')
    print('- The display in the app might be truncating or filtering the list')
    
    print('\nRecommended fixes:')
    print('1. Normalize team names (uppercase/lowercase) in get_all_matchups')
    print('2. Add a search feature in the app to find teams by substring')
    print('3. Create a way to manually add matchups that might be missing')
    
except Exception as e:
    print(f'Error during debugging: {e}')
