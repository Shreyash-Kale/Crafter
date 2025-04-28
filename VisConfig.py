# ---- VisConfig.py ----
import os

# Get the absolute path to the Crafter directory
CRAFTER_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths for results and logs
DEFAULT_LOG_DIR = os.path.join(CRAFTER_DIR, 'logs')
RESULTS_LOG_DIR = os.path.join(CRAFTER_DIR, 'results')

# Debug prints to help troubleshoot
print(f"Looking for logs in: {DEFAULT_LOG_DIR}")
print(f"Looking for results in: {RESULTS_LOG_DIR}")

# Visualization settings
VIZ_COLORS = {
    'cumulative': '#1f77b4',  # Blue for overall progress
    'reward': '#ff7f0e',      # Orange for immediate rewards
    'position': '#d62728',    # Red for position marker
    'significant': '#2ca02c', # Green for significant events
    # Health and resource colors matching game
    'health': '#d62728',      # Red
    'food': '#2ca02c',        # Green
    'drink': '#1f77b4',       # Blue
    'energy': '#ffbf00',      # Golden
    'wood': '#8c564b',        # Brown
    'stone': '#7f7f7f',       # Grey
    'diamond': '#17becf',     # Cyan
    'iron': '#e377c2'         # Pink
}

# Video player settings
DEFAULT_FPS = 30

# Create required directories if they don't exist
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_LOG_DIR, exist_ok=True)
