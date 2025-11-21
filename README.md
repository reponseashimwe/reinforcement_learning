# Dermatology Clinic Triage - Reinforcement Learning

**Author:** [Your Name]
**Date:** November 2025
**Course:** Reinforcement Learning Summative Assignment

---

## 📋 Project Overview

This project implements and compares four reinforcement learning algorithms for optimizing patient triage in a dermatology clinic setting. The agent learns to make optimal decisions about patient routing, resource allocation, and queue management to maximize correct triages while minimizing wait times.

### Problem Statement

In a busy dermatology clinic, patients arrive with varying severity levels (mild, moderate, severe, critical). The agent must:
- **Correctly triage** patients to appropriate care levels (remote advice, nurse, doctor, or urgent escalation)
- **Manage resources** (exam rooms) efficiently
- **Minimize patient wait times**
- **Balance** exploration vs. exploitation to learn optimal policies

### Algorithms Implemented

1. **DQN** (Deep Q-Network) - Value-based method
2. **PPO** (Proximal Policy Optimization) - Policy gradient method
3. **A2C** (Advantage Actor-Critic) - Policy gradient method
4. **REINFORCE** (Vanilla Policy Gradient) - Custom PyTorch implementation

---

## 🏗️ Project Structure

```
reinforcement_learning/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py              # ClinicEnv (Gymnasium)
│   └── rendering.py               # Pygame visualization
│
├── training/
│   ├── dqn_training.py            # DQN training script
│   ├── ppo_training.py            # PPO training script
│   ├── a2c_training.py            # A2C training script
│   └── reinforce_training.py      # Custom REINFORCE implementation
│
├── notebooks/                      # Colab experiment notebooks
│   ├── 00_env_test.ipynb
│   ├── 01_dqn_experiments.ipynb
│   ├── 02_ppo_experiments.ipynb
│   ├── 03_a2c_experiments.ipynb
│   └── 04_reinforce_experiments.ipynb
│
├── configs/                        # Hyperparameter configurations
│   ├── dqn_configs.json           # 10 DQN configurations
│   ├── ppo_configs.json           # 10 PPO configurations
│   ├── a2c_configs.json           # 10 A2C configurations
│   └── reinforce_configs.json     # 10 REINFORCE configurations
│
├── models/                         # Saved trained models
│   ├── dqn/
│   ├── ppo/
│   ├── a2c/
│   └── reinforce/
│
├── logs/                           # Training logs and results
│   ├── dqn/
│   ├── ppo/
│   ├── a2c/
│   └── reinforce/
│
├── evaluation/
│   ├── aggregate_results.py       # Results aggregation script
│   ├── results_summary.csv        # Final comparison table
│   └── plots/                     # Generated plots
│
├── demos/
│   ├── random_demo.mp4            # Random agent video
│   └── best_agent_demo.mp4        # Trained agent video
│
├── main.py                         # Best model entry point
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[yourname]/[yourname]_rl_summative.git
   cd [yourname]_rl_summative
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Best Model

```bash
# Run best PPO model (example)
python main.py --model_type ppo --model_path models/ppo/best_model.zip --run_demo

# Evaluate DQN model
python main.py --model_type dqn --model_path models/dqn/best_model.zip --num_eval_episodes 50

# Generate demo video
python main.py --model_type ppo --model_path models/ppo/best_model.zip --save_video demos/ppo_demo.mp4
```

### Training Models

**Using Google Colab (Recommended):**
1. Upload the project to Google Drive
2. Open one of the experiment notebooks (e.g., `notebooks/02_ppo_experiments.ipynb`)
3. Run all cells sequentially
4. Models and results will be saved to your Google Drive

**Using Local Machine:**
```bash
# Example: Train PPO with specific config
python training/ppo_training.py --config configs/ppo_configs.json --config_id ppo_baseline
```

---

## 🎯 Environment Details

### ClinicEnv Specification

**Observation Space (15 dimensions):**
- `[0]` Patient age (normalized)
- `[1]` Symptom duration (normalized)
- `[2]` Fever flag (binary)
- `[3]` Infection flag (binary)
- `[4-11]` Symptom embedding (8-dim vector)
- `[12]` Room availability flag
- `[13]` Queue length (normalized)
- `[14]` Time of day / episode progress (normalized)

**Action Space (8 discrete actions):**
0. **Send to Doctor** - Direct to dermatologist
1. **Send to Nurse** - Route to nurse practitioner
2. **Remote Advice** - Provide telemedicine consultation
3. **Escalate Priority** - Mark as urgent (for critical cases)
4. **Defer Patient** - Postpone to end of queue
5. **Idle** - Wait / no action
6. **Open Room** - Open additional exam room
7. **Close Room** - Close an exam room

**Reward Structure:**
- **Correct Triage:**
  - Mild → Remote: +1.0
  - Moderate → Nurse: +1.0
  - Severe → Doctor: +2.0
  - Critical → Escalate (fast): +3.0

- **Penalties:**
  - Incorrect triage: -1.5
  - Wait time: -0.01 × queue_size (per step)
  - Resource cost: -0.05 × num_open_rooms (per step)

**Severity Distribution:**
- Mild (40%): Minor rash, low symptom scores
- Moderate (35%): Moderate symptoms, may need exam
- Severe (20%): Suspicious lesions, infection signs
- Critical (5%): Urgent cases requiring immediate attention

---

## 🔬 Experimental Setup

### Hyperparameter Tuning

Each algorithm was tested with **10 different hyperparameter configurations** exploring:

**DQN:**
- Learning rates: 0.0001 - 0.001
- Gamma: 0.98 - 0.995
- Buffer sizes: 30,000 - 100,000
- Batch sizes: 32 - 128
- Exploration schedules
- Target update frequencies

**PPO:**
- Learning rates: 0.0001 - 0.0005
- N-steps: 256 - 512
- Clip ranges: 0.1 - 0.3
- Entropy coefficients: 0.0 - 0.05
- GAE lambda: 0.95 - 0.98

**A2C:**
- Learning rates: 0.0003 - 0.001
- N-steps: 3 - 16
- Entropy coefficients: 0.0 - 0.05
- Value function coefficients: 0.5 - 1.0

**REINFORCE:**
- Learning rates: 0.0003 - 0.003
- Hidden dimensions: [64, 64] - [256, 256]
- With/without baseline
- Entropy coefficients: 0.0 - 0.05

### Training Procedure

**Phase 1: Quick Sweep (Config Selection)**
- Train each config for **50,000 timesteps**
- Evaluate on validation seeds
- Select top 1-2 configs per algorithm

**Phase 2: Full Training (Best Configs)**
- Train for **300,000 - 500,000 timesteps**
- Use **5 random seeds** for statistical significance
- Track comprehensive metrics:
  - Episode rewards
  - Triage accuracy
  - Wait times
  - Queue lengths
  - Convergence speed

---

## 📊 Evaluation Metrics

### Primary Metrics
1. **Mean Episode Reward** - Overall performance
2. **Triage Accuracy** - % of correct triage decisions
3. **Average Wait Time** - Patient queue wait time
4. **Steps to Convergence** - Learning speed
5. **Training Stability** - Reward variance

### Evaluation Protocol
- **Training set:** Seeds 0-4
- **Validation set:** Seeds 5-9
- **Test set:** Seeds 10-19
- **Episodes per evaluation:** 50
- **Deterministic policy** during evaluation

---

## 📈 Results Summary

*(To be filled after experiments)*

### Best Performing Configurations

| Algorithm | Config ID | Mean Reward | Triage Accuracy | Convergence Episodes |
|-----------|-----------|-------------|-----------------|---------------------|
| DQN       | TBD       | TBD         | TBD             | TBD                 |
| PPO       | TBD       | TBD         | TBD             | TBD                 |
| A2C       | TBD       | TBD         | TBD             | TBD                 |
| REINFORCE | TBD       | TBD         | TBD             | TBD                 |

### Key Findings

*(To be filled after analysis)*

---

## 🎥 Demonstrations

- **Random Agent:** `demos/random_demo.mp4`
- **Best Agent:** `demos/best_agent_demo.mp4`

---

## 📚 References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
2. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv*.
3. Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML*.
4. Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*.
5. Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
6. Gymnasium Documentation: https://gymnasium.farama.org/

---

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

### Adding New Configurations
Edit the JSON files in `configs/` directory and add new configuration dictionaries.

---

## 📝 License

This project is created for academic purposes as part of the ALU Reinforcement Learning course.

---

## 👤 Author

**[Your Name]**
Email: [your.email@alustudent.com]
GitHub: [https://github.com/yourname]
Video Demonstration: [Link to 3-minute video]

---

## 🙏 Acknowledgments

- ALU Faculty for project guidance
- Stable-Baselines3 team for RL implementations
- Gymnasium community for environment framework
