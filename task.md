# Overview

This assignment aims to ensure students use their missions to train a reinforcement learning agent by comparing RL Methods, that is, Value-Based (Deep Q Networks) and Policy Methods (REINFORCE, PPO, A2C), to optimize for a simulated mission-based environment.

# Tasks

1. Develop a non-generic environment based on your potential capstone project/mission.
2. Implement a custom environment. To assist you in implementing this:
   - Define Action space: Ensure the list of actions is exhaustive and relevant to your use case.
   - Observation Space.
   - Rewards are associated with states and/or actions.
   - Start State.
   - Terminal Conditions.
3. Visualize the environment using an advanced simulation library such as Unity, pygame, or OpenGL.
4. Create a static file that shows the agent taking random actions (not using a model) in the custom environment. This demonstrates component visualization without training.
5. Add a diagram of the agent in a simulated environment with proper descriptions.
6. Implement and train four separate RL models using the Stable Baselines library in Python, all sharing the same environment for comparison:
   - Value Based: DQN (Value-Based).
   - Policy Gradient Methods: REINFORCE algorithm, Proximal Policy Optimization (PPO), Actor-Critic (A2C).
7. Tweak hyperparameters extensively and discuss observed behavior. Use at least 10 runs with different hyperparameter combinations for each algorithm.
8. Record a video simulating the agent in action:
   - Share your entire screen with the camera on.
   - State the problem briefly.
   - Describe agent behavior.
   - Explain the reward structure.
   - State the objective of the agent.
   - Run the simulation with your best-performing agent, showing the GUI and terminal verbose outputs.
   - Explain agent performance in the simulation.
9. Document results in a PDF report using the provided template. Focus on concise explanations.
10. Provide a `requirements.txt` since the project will be cloned and executed.
11. Create a GitHub repository named `student_name_rl_summative` with the following structure:

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Visualization GUI components
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO/other PG using SB3
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for running best performing model 
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

# Submission Instructions

- Submit a PDF report to Canvas.
- Avoid generic use cases; they will be heavily penalized.

# Evaluation Criteria Highlights

- **Environment Validity & Complexity:** Well-structured action space, rewards, and termination conditions. Agent must explore all actions, including edge cases.
- **Policy Training and Performance:** Demonstrate agent performance with metrics such as average reward, steps per episode, and convergence time. Balance exploration and exploitation; discuss weaknesses and improvements.
- **Visualization:** Use high-quality 2D/3D visualization (OpenGL, Panda3D, Gym, PyBullet). Provide interactive, clear feedback of agent state.
- **Stable Baselines/Policy Gradient Implementation:** Implement policy gradient with well-tuned hyperparameters and justification.
- **Discussion & Analysis:** Provide clear, labeled graphs and precise descriptions linking visuals to metrics. Combine qualitative insights with numerical evidence and creative visualization where appropriate.

# Report Template Summary

## Reinforcement Learning Summative Assignment Report

- **Student Name:** [Your Name]
- **Video Recording:** [Link to video (≤3 min, camera on, full screen)]
- **GitHub Repository:** [Link to `student_name_rl_summative`]

### Project Overview
Brief paragraph (≈5 lines) describing the problem, approach, and RL implementation purpose.

### Environment Description
- **Agent(s):** Describe behavior, representation, and capabilities.
- **Action Space:** List discrete/continuous actions available.
- **Observation Space:** Detail observations provided and encoding.
- **Reward Structure:** Explain reward function with formulas if applicable.

### Environment Visualization
Include a 30-second video demonstrating the visualization and explain elements.

### System Analysis and Design
- **Deep Q-Network (DQN):** Describe architecture, target network, replay buffer, modifications.
- **Policy Gradient Method (REINFORCE/PPO/A2C):** Describe architecture, policy representation, special features.

### Implementation Logs
Provide tables (≥10 rows each) detailing hyperparameter sweeps and results for:
- DQN (learning rate, gamma, replay buffer size, batch size, exploration strategy, mean reward, etc.).
- REINFORCE (learning rate and additional relevant columns).
- A2C (learning rate and additional relevant columns).
- PPO (learning rate and additional relevant columns).

### Results Discussion
- **Visualizations:** Describe and interpret every plot.
- **Cumulative Rewards:** Plot subplots of cumulative rewards for best models across methods.
- **Training Stability:** Show objective function curves for DQN and policy entropy for policy gradient; analyze stability.
- **Episodes to Converge:** Quantify episodes required for stable performance with plots.
- **Generalization:** Report testing on unseen initial states and analyze generalization.

### Conclusion and Discussion
Summarize findings, compare method performance, note strengths/weaknesses, and suggest improvements.

# Additional Requirements

- Record a video of the agent simulation as described.
- Ensure all files (code, visualization, static random-action demo, diagram, report) are organized and reproducible.
- Avoid referencing the provided example scenario in the final submission.
