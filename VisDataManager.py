# data_manager.py - Handles data loading and processing

import pandas as pd
import numpy as np
import os

class DataManager:
    """Manages loading and processing data from event logs"""
    def get_ppo_entropy_norm(self): return self.ppo_entropy_norm
    def get_ppo_advantage_norm(self): return self.ppo_advantage_norm
    def get_dreamer_explore_norm(self): return self.dreamer_explore_norm
    def get_dreamer_wm_score_norm(self): return self.dreamer_wm_score_norm
    def get_value_norm(self):          return self.value_norm
    def get_action_prob_norm(self):    return self.action_prob_norm




    def get_action_name(self, action):
        """Convert action ID to action name - refer https://arxiv.org/pdf/2109.06780.pdf for details"""
        ACTION_MAPPING = {
            0: "noop",
            1: "move_left",
            2: "move_right",
            3: "move_up",
            4: "move_down",
            5: "do",
            6: "sleep",
            7: "place_stone",
            8: "place_table",
            9: "place_furnace",
            10: "place_plant",
            11: "make_wood_pickaxe",
            12: "make_stone_pickaxe",
            13: "make_iron_pickaxe",
            14: "make_wood_sword",
            15: "make_stone_sword",
            16: "make_iron_sword"
        }
        
        if action is None:
            return "unknown"
        
        # If action is already a string, check if it's a valid action name
        if isinstance(action, str):
            # Check if it's one of our known action names
            if action.lower() in ACTION_MAPPING.values():
                return action.lower()
            # Try to convert string to integer (if it's a string like "6")
            try:
                action_id = int(action)
                return ACTION_MAPPING.get(action_id, f"Unknown ({action_id})")
            except ValueError:
                print(f"Cannot convert action '{action}' to integer. Returning original string.")
                return action  # Return the original string if it can't be converted
        
        # Handle numeric action IDs
        return ACTION_MAPPING.get(action, f"Unknown ({action})")



    
    def __init__(self):
        # Initialize empty data containers
        self.event_df = None
        self.time_steps = []
        self.reward_log = []
        self.action_log = []
        self.action_prob = []
        self.value_est = []

        self.ppo_entropy = []
        self.ppo_advantage = []
        self.ppo_entropy_norm = []
        self.ppo_advantage_norm = []
        self.action_prob_norm = []
        self.value_norm       = []


        self.dreamer_explore = []
        self.dreamer_wm_score = []
        self.dreamer_explore_norm = []
        self.dreamer_wm_score_norm = []
        self.reward_components = {}

        # ── decision-attribution placeholders (so no AttributeError) ──
        self.action_prob = self.value_est = []
        self.ppo_entropy = self.ppo_advantage = []
        self.dreamer_explore = self.dreamer_wm_score = []

        # normalised versions start as empty arrays
        self.ppo_entropy_norm = self.ppo_advantage_norm = []
        self.dreamer_explore_norm = self.dreamer_wm_score_norm = []
        

    
    def load_data(self, csv_path):
        """Load data from a CSV file (supports both legacy and new formats)"""
        # Initialize/reset data containers
        self.event_df = None
        self.time_steps = []
        self.reward_log = []
        self.action_log = []
        self.executed_action_log = []
        self.reward_components = {}
        
        self.achievement_dependencies = {
            'collect_diamond': ['make_iron_pickaxe'],
            'make_iron_pickaxe': ['collect_iron', 'place_table'],
            'make_iron_sword': ['collect_iron', 'place_table'],
            'make_stone_pickaxe': ['collect_stone', 'place_table'],
            'make_stone_sword': ['collect_stone', 'place_table'],
            'make_wood_pickaxe': ['collect_wood', 'place_table'],
            'make_wood_sword': ['collect_wood', 'place_table'],
            'place_furnace': ['collect_stone'],
            'place_table': ['collect_wood']
        }
        
        try:
            # Read the CSV file into a DataFrame
            self.event_df = pd.read_csv(csv_path)
            
            # Extract basic trajectory information first
            self.time_steps = self.event_df['time_step'].tolist()
            self.reward_log = self.event_df['reward'].tolist()
            self.action_log = self.event_df['action'].tolist()
            # Add decision-attribution columns
            # self.action_prob = self.event_df.get("action_prob", []).tolist()
            # self.value_est   = self.event_df.get("value", self.event_df.get("value_estimate", [])).tolist()

            def _safe_list(val):
                """Return a plain list whether val is a Series or already a list."""
                return val.tolist() if hasattr(val, "tolist") else val

            self.action_prob   = _safe_list(self.event_df.get('action_probability', []))
            self.value_est     = _safe_list(self.event_df.get('value', self.event_df.get('value_estimate', [])))

            # PPO extras
            self.ppo_entropy   = _safe_list(self.event_df.get('entropy', []))
            self.ppo_advantage = _safe_list(self.event_df.get('advantage', []))

            # Dreamer extras
            self.dreamer_explore  = _safe_list(self.event_df.get('exploration_bonus', []))
            self.dreamer_wm_score = _safe_list(self.event_df.get('world_model_score', []))


            def _norm(arr):
                arr = np.asarray(arr, dtype=np.float32)
                if arr.size == 0 or np.max(arr) == np.min(arr):
                    return np.zeros_like(arr)
                return (arr - arr.min()) / (arr.max() - arr.min())


            self.ppo_entropy_norm      = _norm(self.ppo_entropy)
            self.ppo_advantage_norm    = _norm(self.ppo_advantage)
            self.dreamer_explore_norm  = _norm(self.dreamer_explore)
            self.dreamer_wm_score_norm = _norm(self.dreamer_wm_score)
            self.action_prob_norm = _norm(self.action_prob)
            self.value_norm       = _norm(self.value_est)


            
            # Check for executed_action column
            if 'executed_action' in self.event_df.columns:
                self.executed_action_log = self.event_df['executed_action'].tolist()
            else:
                # Create a copy to avoid reference issues
                self.executed_action_log = self.action_log.copy()
            
            # Extract reward components (handling different CSV formats)
            exclude_cols = ['time_step', 'action', 'reward', 'cumulative_reward', 
                        'inventory', 'discount', 'semantic', 'player_pos']
            
            # Build reward components dictionary
            self.reward_components = {}
            
            # Handle the new 'data.csv' format - extract inventory items as components
            if 'inventory' in self.event_df.columns:
                try:
                    # First pass: identify all possible inventory keys
                    inventory_keys = set()
                    for inventory_str in self.event_df['inventory']:
                        if isinstance(inventory_str, str):
                            try:
                                inventory = eval(inventory_str)  # Convert string to dict
                                inventory_keys.update(inventory.keys())
                            except:
                                pass  # Skip invalid entries
                    
                    # Initialize arrays for all keys
                    for key in inventory_keys:
                        self.reward_components[key] = [0] * len(self.time_steps)
                    
                    # Second pass: populate values
                    for i, inventory_str in enumerate(self.event_df['inventory']):
                        if isinstance(inventory_str, str):
                            try:
                                inventory = eval(inventory_str)
                                for key, value in inventory.items():
                                    if key in self.reward_components:
                                        self.reward_components[key][i] = value
                            except:
                                pass  # Skip invalid entries
                except Exception as e:
                    print(f"Error parsing inventory data: {e}")
            
            # Handle traditional format - look for achievement/component columns
            component_cols = [col for col in self.event_df.columns if col not in exclude_cols]
            
            for col in component_cols:
                # Only include components that have non-zero values
                values = self.event_df[col].tolist()
                if any(v != 0 for v in values):
                    self.reward_components[col] = values
                    
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def get_completed_achievements(self, step=None):
        """Get a list of completed achievements based on reward components up to a specific step"""
        achievement_list = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        completed = []
        for ach in achievement_list:
            if ach in self.reward_components:
                # If step is provided, only check up to that step
                if step is not None:
                    # Make sure we don't exceed array bounds
                    max_step = min(step + 1, len(self.reward_components[ach]))
                    values = self.reward_components[ach][:max_step]
                    if any(v != 0 for v in values):
                        completed.append(ach)
                else:
                    # Original behavior (check all steps)
                    if any(v != 0 for v in self.reward_components[ach]):
                        completed.append(ach)
        
        return completed


    def is_achievement_completed(self, achievement):
        """Check if an achievement is completed based on reward components"""
        if achievement in self.reward_components:
            return any(v != 0 for v in self.reward_components[achievement])
        return False

    def get_available_achievements(self):
        """Get a list of achievements that are available but not completed"""
        available = []
        completed = self.get_completed_achievements()
        
        for ach, deps in self.achievement_dependencies.items():
            if ach not in completed:  # If not already completed
                if all(dep in completed for dep in deps):
                    available.append(ach)
        
        # Also include achievements with no dependencies that aren't completed
        achievement_list = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        for ach in achievement_list:
            if ach not in completed and ach not in self.achievement_dependencies:
                available.append(ach)
        
        return available

    def get_achievement_dependencies(self, achievement):
        """Get dependencies for an achievement"""
        return self.achievement_dependencies.get(achievement, [])

    def get_step_achievements(self, step):
        """Get achievements that were completed at a specific step"""
        achievement_list = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        step_achievements = []
        for ach in achievement_list:
            if ach in self.reward_components and step < len(self.reward_components[ach]):
                if self.reward_components[ach][step] != 0:
                    step_achievements.append(ach)
        
        return step_achievements

  
    
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

