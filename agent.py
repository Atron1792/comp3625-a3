"""
The only change from Derek's lab: the starting weights are handed in instead of being hard-coded, 
and learning can be switched off (alpha=0)
so we can measure innate, pre-learning performance.
"""

import random

import numpy as np

from gridworld import GridWorld

ACTIONS = ["up", "right", "down", "left"]
N_FEATURES = 13 # 12 sensory inputs + 1 bias
GENOME_SIZE = len(ACTIONS) * N_FEATURES

GAMMA = 0.9

def state_to_features(state: dict) -> np.ndarray:
    """Convert the sensory dict into a feature vector. Index 0 is a constant bias term of 1."""
    keys = sorted(state.keys())  # sorted so the features are always in the same order
    values = [float(state[k]) for k in keys]
    return np.array([1.0] + values)

def predict_q_values(weights: dict, features: np.ndarray, actions: list) -> dict:
    """Q(s, a) for every action: the dot product of that action's weights and the features."""
    return {a: float(np.dot(weights[a], features)) for a in actions}

def choose_action(q_vals: dict, actions: list, epsilon: float) -> str:
    """Epsilon-greedy: explore with probability epsilon, otherwise pick the best action."""
    if random.random() < epsilon:
        return random.choice(actions)
    best_q = max(q_vals.values())
    best_actions = [a for a in actions if q_vals[a] == best_q]
    return random.choice(best_actions)  # break ties randomly


def feature_names() -> list:
    """
    Names for the 13 feature columns, in the same order as state_to_features
    ('bias', then the sorted sensory keys like 'hazard_right').
    Used by analysis.py to label weights.
    """
    pass

def genome_to_weights(genome: np.ndarray) -> dict:
    """Reshape a flat genome (GENOME_SIZE,) into {action: weight vector (N_FEATURES,)}."""
    pass

def weights_to_genome(weights: dict) -> np.ndarray:
    """Inverse of genome_to_weights: flatten {action: vector} into one array in ACTIONS order."""
    pass

def live(world: GridWorld, genome: np.ndarray, n_steps: int, alpha: float, epsilon: float):
    """
    Live one lifetime in the world, starting from the genome's weights.

    The genome itself is not modified (learning happens on a copy). Copy holds what is learnt

    :param alpha: learning rate. alpha=0 means no learning (innate behaviour only).
    :param epsilon: exploration rate (keep it simple: fixed value, no schedule)
    :return: (final_weights as a flat array, total_reward over the lifetime)
    """
    pass
