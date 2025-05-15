# Crafter Project – Visual Demo & Agent Analysis

This project includes training, evaluation, and visualization modules for analyzing reinforcement learning agents in the [Crafter environment](https://github.com/danijar/crafter). The agents used include **DreamerV2** and **PPO**, with a custom-built visualization UI to explore behaviors and decisions.

---

## 📽️ Demo of Visualization UI

> Explore how the custom-built UI visualizes reward signals, action trajectories, and state transitions.

📎 [Click here to watch the full demo video](https://drive.google.com/file/d/1GvucbAHuXWmhE8PJqDv-3kwOFSbceZ2B/view)

[![Demo Visualization](assets/Viz_Tool.png)](https://drive.google.com/file/d/1GvucbAHuXWmhE8PJqDv-3kwOFSbceZ2B/view)

*(Click the image above to open the demo video hosted on Google Drive)*

---

## 📊 Selected Plot Outputs

Visualizations from both Dreamer and PPO runs:

| Plot | Description |
|------|-------------|
| ![](assets/Agent%20Reward%20Timeline.png) | **Reward Timeline**: Cumulative reward over time |
| ![](assets/Action%20Probability.png) | **Action Probabilities**: Policy’s action distribution |
| ![](assets/Exploration%20Bonus.png) | **Exploration Bonus**: Internal motivation to explore |
| ![](assets/Reward%20Decomposition%20Breakdown.png) | **Decomposed Rewards**: Contributions from each sub-reward |
| ![](assets/Valuye%20Estimate.png) | **Value Estimate**: Predicted return at each step |
| ![](assets/World-Model%20Score.png) | **World Model Confidence**: Quality of latent model predictions |

---

# ⚙️ Running the Modules

This repo includes three main modules:

- **Visualization** (`viz/`)
- **DreamerV2 agent** (`dreamer/`)
- **PPO agent** (`ppo/`)


---

## 🚀 Quick Start

| Task                    | Script Path                | Command                             |
|-------------------------|----------------------------|-------------------------------------|
| Visualization UI        | `viz/VisMain.py`           | `python viz/VisMain.py`             |
| Dreamer Training        | `dreamer/dreamer_train.py` | `python dreamer/dreamer_train.py`   |
| Dreamer Evaluation      | `dreamer/environment.py`   | `python dreamer/environment.py`     |
| PPO Training            | `ppo/train_ppo_crafter.py` | `python ppo/train_ppo_crafter.py`   |
| PPO Evaluation          | `ppo/eval_ppo_crafter.py`  | `python ppo/eval_ppo_crafter.py`    |

---

# 📂 Running the Visualization Module with Sample Data

To run the visualization module:

1. Execute the visualization script:
   ```bash
   python viz/VisMain.py
   ```

2. Ensure that both the CSV log files and the corresponding MP4 video files are present in their respective folders. These are required for the UI to display synchronized agent behavior and reward signal visualizations.

3. Sample log and video files are provided in the repository for both Dreamer and PPO agents:

   - **DreamerV2:**
     - Log file: `Plots_Dreamer/ckpt270000_episode300.csv`
     - Video file: `Plots_Dreamer/ckpt270000_episode300.mp4`
   
   - **PPO:**
     - Log file: `Plots_PPO/episode_295.csv`
     - Video file: `Plots_PPO/episode_295.mp4`

These files are pre-integrated to allow immediate testing of the visualization module.

> If you wish to generate your own evaluation logs and gameplay videos:

- For **DreamerV2**, run:
  ```bash
  python dreamer/environment.py
  ```

- For **PPO**, run:
  ```bash
  python ppo/eval_ppo_crafter.py
  ```

Generated outputs can be saved and reused with the visualization UI by placing them in the correct folders or updating the script paths as needed.

---

## 🔧 Prerequisites

Before starting:

1. Activate your Python virtual environment (if applicable).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Notes

1. ⁠*Model Checkpoints:*  
  Always ensure that you have a trained model checkpoint before attempting to run any evaluation script.

2. ⁠*Custom Paths:*  
  Paths for saving models, logs, or results can be adjusted inside each script depending on your project organization.
