import numpy as np
import matplotlib.pyplot as plt


def mean_learned_delta(log: dict, generations: slice) -> np.ndarray:
    """Average learned genome over the given generations and all individuals. Shape (GENOME_SIZE,)."""
    pass

def genome_drift(log: dict) -> np.ndarray:
    """Mean genome at each generation minus mean genome at generation 0. Shape (n_gen, GENOME_SIZE)."""
    pass

def cosine_similarity_over_time(drift: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cosine similarity between each generation's drift and a fixed target vector. Shape (n_gen,)."""
    pass


# figures

def plot_fitness(log_learn: dict, log_ga_only: dict):
    """
    The mean fitness per generation for GA plus learning, GA-only, and
    the innate (no-learning) fitness of the GA plus learning population.
    Shows GA+learning > GA-only, and innate performance rising over generations.
    """
    pass


def plot_assimilation(log_learn: dict, log_ga_only: dict, early_gens: slice = slice(0, 5)):
    """
    Cosine similarity between genome drift and the early learned delta,
    over generations, for both conditions. The Baldwin signature is the
    GA+learning line rising while GA-only stays near zero.
    """
    pass


def plot_weight_traces(log_learn: dict, log_ga_only: dict, picks: list):
    """
    Mean genome value over generations for a few hand-picked weights
    (e.g. ('right', 'hazard_right'), ('down', 'goal_down')), next to the value
    learned during life in early generations. 

    :param picks: list of (action, feature_name) tuples
    """
    pass
