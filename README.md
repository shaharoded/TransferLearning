# Deep Reinforcement Learning - Assignment 3

## Training and Comparing Meta-Learning and Transfer-Learning using an Actor-Critic agent

## Introduction

Deep reinforcement learning algorithms usually require a large number of trials, derived from the need to re-collect large dataset and training from scratch per task. Intuitively, knowledge gained in learning one task should help to learn new, related tasks more quickly. Humans and animals are able to learn new tasks in just a few trials. Here we designed a reinforcement learning algorithm that leverages prior experience to figure out how to solve new tasks quickly. In recent literature, such methods are referred to as meta-reinforcement learning [Mishra et al. 2018, Finn et al. 2017, Wang et al. 2016, Duan et al. 2016].

## Assignment Structure

### Section 1: Training Individual Networks
In this section we implemented a one-step Actor-Critic algorithm on three control problems: 
1. CartPole-v1
2. Acrobot-v1
3. MountainCarContinuous-v0

**Technical Implementation:**
- **Unified Architecture**: All tasks use identical network dimensions (state_dim=6, action_dim=5) with input padding for smaller observation spaces and action masking for unused actions
- **MountainCar Discretization**: Converted continuous action space to 5 discrete actions using custom adapters
- **Reward Shaping**: Implemented a custom wrapper for MountainCar providing intermediate rewards:
  - Position bonuses (weight=10.0) for reaching new heights
  - Velocity bonuses (weight=1.0) for building momentum  
  - Large goal completion bonus (+100) for reaching the flag
- **Hyperparameter Tuning**: Used entropy_coef=0.10 for MountainCar (vs 0.01 for others) to reduce exploration after discovering successful strategies
- **Success Criteria**: Defined task-specific success metrics:
  - **CartPole**: Episode length ≥ 500 steps (full episode completion)  
    → Similar difficulty to $MA_{100 episodes}$ ≥ 475 demand
  - **Acrobot**: Episode reward ≥ -100 (standard threshold)
  - **MountainCar**: Episode length < 999 steps (reaching flag before timeout)
- **Convergence Definition**: Target 95% success rate over moving average window of 100 episodes
- **Model Persistence**: Automatic saving of converged models using environment names (e.g., `models/CartPole-v1.pth`)

All agents converged within 1000-2000 episodes, demonstrating effective learning across diverse control tasks.

<img src="results/part1_plots.png" alt="Agents training Process" width="700"/>


### Section 2: Fine-Tuning an Existing Model (25%)
Fine-tuned models trained on one problem to apply to another. For pairs: Acrobot → CartPole, CartPole → MountainCar:

**Methodology:**
- Loaded fully trained source models from Section 1
- Re-initialized output layer weights for both actor and critic networks (found to work better than actor-only re-initialization)
- Maintained source model's hidden layer weights to preserve learned features
- Trained on target tasks with same hyperparameters as Section 1

**Results:**
- **Convergence Time**: Remained similar to Section 1 baselines
- **Stability**: Fine-tuned models showed steadier learning curves with less variance
- **Performance**: Re-initializing both actor and critic output layers provided better results than actor-only re-initialization
- **Comparison**: Fine-tuning did not significantly accelerate convergence but provided more stable training

The results suggest that while fine-tuning preserves some useful features from source tasks, the architectural differences between control problems may limit transfer benefits.

<img src="results/part2_cartpole_plots.png" alt="Cartpole Training Comparison" width="700"/>

<img src="results/part2_mountaincar_plots.png" alt="Mountaincar Training Comparison" width="700"/>

### Section 3: Transfer Learning with Progressive Networks
Implemented a simplified Progressive Networks approach for multi-source transfer learning.

**Scenarios:**
- **Scenario A**: {Acrobot, MountainCar} → CartPole
- **Scenario B**: {CartPole, Acrobot} → MountainCar

**Architecture:**
Progressive Networks leverage knowledge from multiple pre-trained source models by:
1. **Freezing Source Networks**: All source model parameters remain fixed during training
2. **Feature Extraction**: Input states pass through all source models' hidden layers
3. **Lateral Connections**: Source features are concatenated with target features
4. **New Output Heads**: Fresh actor/critic heads trained on the fused representation

```
Input State (x)
    │
    ├──► Source 1 Hidden (frozen) ──► h₁
    ├──► Source 2 Hidden (frozen) ──► h₂
    └──► Target Hidden (trainable) ─► h_target
                                        │
                    ┌───────────────────┴───────────────────┐
                    │  Concatenate: [h_target, h₁, h₂, ...]  │
                    └───────────────────┬───────────────────┘
                                        │
                            ┌───────────┴───────────┐
                            │                       │
                    Actor Head (new)         Critic Head (new)
```

**Technical Implementation:**
- **Unified Input Handling**: Zero-padding applied for environments with smaller state dimensions
- **Fused Feature Dimension**: Target hidden (128) + Source1 hidden (128) + Source2 hidden (128) = 384D
- **Trainable Parameters**: Only target trunk and new output heads are optimized
- **Same Convergence Criteria**: 95% success rate over 100-episode moving window

**Results:**

<img src="results/part3_cartpole_plots.png" alt="Progressive CartPole Training Comparison" width="700"/>

<img src="results/part3_mountaincar_plots.png" alt="Progressive MountainCar Training Comparison" width="700"/>

<img src="results/part3_diagnostics.png" alt="Progressive Networks Diagnostics" width="700"/>

**Analysis:**
- Progressive Networks provide richer initial representations from multiple source tasks
- The frozen source features act as complementary "perspectives" on the input state
- Avoids catastrophic forgetting since source weights are never modified
- Performance depends on similarity between source and target task dynamics

## Project Structure
```
Root/
├── src/
│   ├── adapters.py                     # Environment adapters for unified I/O and action discretization
│   ├── agent.py                        # Actor-Critic agent with save/load functionality
│   ├── ffnn.py                         # Feed-forward neural networks (PolicyNetwork, ValueNetwork)
│   ├── progressive.py                  # Progressive Networks for multi-source transfer learning
│   ├── train.py                        # Training functions with success rate convergence
│   ├── training_utils.py               # Utilities for plotting and analysis
│   └── wrappers.py                     # Reward shaping wrappers (MountainCarRewardShaping)
├── mainColab.ipynb                     # Main notebook for training and evaluation
├── results/                            # Training outputs (plots, summaries)
├── models/                             # Saved model checkpoints (auto-saved on convergence)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Algorithms

### Actor-Critic
An online algorithm that updates the policy and value function at every step, rather than at the end of the episode.
- **Update Rule**: Uses the TD-error $\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ as the advantage estimate.
- **Critic**: Learns value function ($V(s)$) using TD learning.
- **Actor**: Updates policy using $\delta_t$ as the critic's evaluation of the action.
- **Entropy Regularization**: Encourages exploration.

### Transfer Learning Methods
- **Fine-Tuning**: Retrain a pre-trained model on a new task by reinitializing output layer.
- **Progressive Networks**: Connect multiple source networks to a target, freezing sources.

## Usage

Open `mainColab.ipynb` to run the training and evaluation pipeline.
The notebook covers:
1. Environment setup and adapters.
2. Training individual networks (Section 1).
3. Fine-tuning models (Section 2).
4. Transfer learning (Section 3).
5. Comparative analysis.
