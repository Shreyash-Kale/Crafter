# config.py - Configuration settings for the application

import os

# Paths
DEFAULT_LOG_DIR = os.path.join(os.getcwd(), 'logs')

# Visualization settings
VIZ_COLORS = {
    'cumulative': '#1f77b4',
    'reward': '#ff7f0e',
    'position': '#d62728',
    'significant': '#2ca02c'
}

# Video player settings
DEFAULT_FPS = 30

# Create required directories if they don't exist
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
