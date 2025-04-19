# data_manager.py - Handles data loading and processing

import pandas as pd
import numpy as np

class DataManager:
    """Manages loading and processing data from event logs"""


    def get_action_name(self, action_id):
        """Convert action ID to human-readable name"""
        ACTION_MAPPING = {
            0: "noop",
            1: "up",
            2: "right",
            3: "down",
            4: "left",
            5: "do",
            6: "sleep",
            7: "attack",
            8: "place_stone",
            9: "place_table", 
            10: "place_furnace",
            11: "place_plant",
            12: "make_wood_pickaxe",
            13: "make_stone_pickaxe",
            14: "make_iron_pickaxe",
            15: "make_wood_sword",
            16: "make_stone_sword",
            17: "make_iron_sword"
        }
        return ACTION_MAPPING.get(action_id, f"Unknown ({action_id})")


    
    def __init__(self):
        # Initialize empty data containers
        self.event_df = None
        self.time_steps = []
        self.reward_log = []
        self.action_log = []
        self.reward_components = {}
    
    def load_data(self, csv_path):
        """Load data from a CSV file"""
        
        try:
            # Read the CSV file into a DataFrame
            self.event_df = pd.read_csv(csv_path)
            
            # Extract basic trajectory information
            self.time_steps = self.event_df['time_step'].tolist()
            self.reward_log = self.event_df['reward'].tolist()
            self.action_log = self.event_df['action'].tolist()
            
            # Extract reward components (all columns except basic info)
            exclude_cols = ['time_step', 'action', 'reward', 'cumulative_reward']
            component_cols = [col for col in self.event_df.columns if col not in exclude_cols]
            
            # Build reward components dictionary
            self.reward_components = {}
            for col in component_cols:
                # Only include components that have non-zero values
                values = self.event_df[col].tolist()
                if any(v != 0 for v in values):
                    self.reward_components[col] = values
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_step_details(self, step):
        """Get detailed information for a specific step"""
        
        if self.event_df is None or step >= len(self.event_df):
            return None
        
        # Get the row for this step
        row = self.event_df.iloc[step]
        
        # Create a dictionary of details
        details = {
            'time_step': row['time_step'],
            'action': row['action'],
            'reward': row['reward'],
            'cumulative_reward': row['cumulative_reward']
        }
        
        # Add all other columns (reward components)
        for col in self.event_df.columns:
            if col not in details:
                details[col] = row[col]
        
        return details
    
    def get_significant_points(self):
        """Identify significant points in the reward sequence"""
        
        if not self.reward_log:
            return []
        
        # Calculate reward changes
        reward_changes = np.diff(self.reward_log, prepend=0)
        
        # Define a threshold for significance (e.g., 1.5 std deviations)
        threshold = np.std(reward_changes) * 1.5
        
        # Find points where change exceeds threshold
        significant_points = np.where(np.abs(reward_changes) > threshold)[0]
        
        return significant_points.tolist()

