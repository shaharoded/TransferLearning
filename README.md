# Deep Reinforcement Learning - Assignment 3

## Training and Comparing Meta-Learning and Transfer-Learning using an Actor-Critic agent

## Introduction

Deep reinforcement learning algorithms usually require a large number of trials. So far with the tools we have learned in this course, learning a new task entails re-collecting this large dataset and training from scratch. Intuitively, knowledge gained in learning one task should help to learn new, related tasks more quickly. Humans and animals are able to learn new tasks in just a few trials. In this assignment, we design a reinforcement learning algorithm that leverages prior experience to figure out how to solve new tasks quickly. In recent literature, such methods are referred to as meta-reinforcement learning [Mishra et al. 2018, Finn et al. 2017, Wang et al. 2016, Duan et al. 2016].

## Assignment Structure

### Section 1: Training Individual Networks (25%)
In this section, implement the actor-critic algorithm from HW2 on three control problems: CartPole-v1, Acrobot-v1, and MountainCarContinuous-v0. Achieve the respective goals: balancing the pole, reaching the mountain top, and bringing the acrobot to a pre-specified height.

To facilitate transfer learning, ensure identical input and output sizes across tasks (pad inputs with 0s for smaller problems, have "empty" actions for smaller outputs).

Notes:
1. Retrain the architecture for CartPole to match sizes.
2. Each architecture must include at least one hidden layer.

- Provide statistics for running time and training iterations required for convergence on each task.

### Section 2: Fine-Tuning an Existing Model (25%)
Fine-tune models trained on one problem to apply to another. For pairs: Acrobot -> CartPole, CartPole -> MountainCar:
- Take the fully trained source model, reinitialize output layer weights.
- Train on the target task.
- Provide statistics on running time and iterations. Compare to Section 1 results. Did fine-tuning lead to faster convergence?

### Section 3: Transfer Learning (50%)
Implement a simplified Progressive Networks approach. For settings: {Acrobot, MountainCar} -> CartPole and {CartPole, Acrobot} -> MountainCar:
- Connect fully-trained source networks to the untrained target network (sources remain frozen).
- Connect hidden layers appropriately (top layers first, then lower if multiple).
- Train until convergence. Did transfer learning improve training? Provide statistics.

Important: Transfer learning can be tricky; document efforts even if not successful.

## Project Structure
```
Root/
├── src/
│   ├── adapters.py                     # Environment adapters for unified I/O
│   ├── agent.py                        # Actor-Critic agent implementation
│   ├── ffnn.py                         # Feed-forward neural networks
│   ├── train.py                        # Training functions
│   ├── training_utils.py               # Utilities for plotting and analysis
│   └── __pycache__/                    # Python cache
├── mainColab.ipynb                     # Main notebook for training and evaluation
├── results/                            # Training outputs (plots, summaries)
├── models/                             # Saved model checkpoints
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Algorithms

### Actor-Critic
The one-step actor-critic algorithm updates policy and value function at every step.
- **Critic**: Learns value function using TD learning.
- **Actor**: Updates policy using TD-error as advantage.
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

## Results
[To be added after experiments]
