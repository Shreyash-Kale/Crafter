# Running Different Modules

This project contains modules for training, evaluation, and visualization of agents in the Crafter environment.

## Quick Start

| Task                         | Script                  | Command                              |
| ----------------------------- | ----------------------- | ------------------------------------ |
| Visualization Module          | ⁠ VisMain.py ⁠             | ⁠ python VisMain.py ⁠                  |
| Dreamer Agent Training         | ⁠ dreamer_train.py ⁠        | ⁠ python dreamer_train.py ⁠             |
| Dreamer Agent Evaluation       | ⁠ environment.py ⁠          | ⁠ python environment.py ⁠              |
| PPO Agent Training             | ⁠ train_ppo_crafter.py ⁠    | ⁠ python train_ppo_crafter.py ⁠         |
| PPO Agent Evaluation           | ⁠ eval_ppo_crafter.py ⁠     | ⁠ python eval_ppo_crafter.py ⁠          |

---

## Detailed Instructions

### Prerequisites

Before running any module:

•⁠  ⁠Activate the correct Python virtual environment (if applicable).
•⁠  ⁠Install all required dependencies:

  ⁠ bash
  pip install -r requirements.txt
   ⁠

•⁠  ⁠Ensure the following directories exist. If they don't, create them:

  ⁠ bash
  mkdir logs results
   ⁠

---

## Running Modules

### Visualization Module

Launch the visualization UI to inspect agent behaviors, rewards, and decision attributions:

⁠ bash
python VisMain.py
 ⁠

•⁠  ⁠Opens an interactive window where you can load episode logs (⁠ .csv ⁠) and corresponding videos (⁠ .mp4 ⁠).

---

### Dreamer Agent Training

Start training the DreamerV2 agent:

⁠ bash
python dreamer_train.py
 ⁠

•⁠  ⁠The training process saves log files under the ⁠ logs directory
•⁠  ⁠Model checkpoints are saved for later evaluation.

---

### Dreamer Agent Evaluation

Evaluate a trained Dreamer agent and generate evaluation results:

⁠ bash
python environment.py
 ⁠

•⁠  ⁠Episode outputs including CSVs and videos are saved in the ⁠ results directory
•⁠  ⁠Ensure a trained Dreamer model checkpoint is available before running this.

---

### PPO Agent Training

Train the PPO agent:

⁠ bash
python train_ppo_crafter.py
 ⁠

•⁠  ⁠The training outputs (model checkpoints and logs) are saved under ⁠ logs ⁠
•⁠  ⁠A ⁠ .zip ⁠ file containing the trained model is also created.

---

### PPO Agent Evaluation

Evaluate a trained PPO agent:

⁠ bash
python eval_ppo_crafter.py
 ⁠

•⁠  ⁠Evaluation outputs, including CSVs and videos, are saved under ⁠ results/ 
•⁠  ⁠Ensure a trained PPO model is available before running this.

---

## Notes

•⁠  ⁠*Model Checkpoints:*  
  Always ensure that you have a trained model checkpoint before attempting to run any evaluation script.

•⁠  ⁠*Custom Paths:*  
  Paths for saving models, logs, or results can be adjusted inside each script depending on your project organization.
