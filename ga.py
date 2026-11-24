import numpy as np
from agent import *
from gridworld import GridWorld

# hyperparameters
POP_SIZE = 50
N_GENERATIONS = 100
LIFESPAN = 500  # steps per lifetime
ALPHA = 0.1 # learning rate during life (0 for GA alone)
EPSILON = 0.1
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1  # per-gene probability
MUTATION_SIGMA = 0.1 # std of gaussian noise
INIT_SIGMA = 0.1  # std for random initial genomes
SOME_RANDOM_SIZE_CHANGE_LATER = 5


def random_genome() -> np.ndarray:
    """Small random weights, shape (GENOME_SIZE,). Gen 0 should know nothing useful."""
    genome = np.random.randn(GENOME_SIZE) * 0.1
    return genome

def evaluate(genome: np.ndarray, alpha: float, n_worlds: int = 1):
    """
    Fitness of one genome: total reward over a lifetime in a fresh randomized world.
    Optionally averaged over n_worlds worlds to reduce noise.

    :return: (fitness, learned_genome) with learned_genome = weights at end of life
    """
    total_fitness = 0.0
    learned_genomes = []

    for _ in range(n_worlds):
        world = GridWorld(size=SOME_RANDOM_SIZE_CHANGE_LATER,
                          state_includes_location=False,
                          state_includes_sensory=True,
                          randomize_each_trial=True)

        # live() copies the genome internally, so the original is never modified
        learned, reward = live(world, genome, LIFESPAN, alpha, EPSILON)

        total_fitness += reward
        learned_genomes.append(learned)

    return total_fitness / n_worlds, np.mean(learned_genomes, axis=0)

    
def select_parent(population: list, fitnesses: np.ndarray) -> np.ndarray:
    """Tournament selection: sample TOURNAMENT_SIZE individuals, return the fittest's genome."""
    pop_size = len(population)
    if num_parents is None:
        num_parents = pop_size

    selected_parents = []

    for _ in range(num_parents):
        # Pick random indices for the tournament participants
        participant_indices = np.random.choice(pop_size, size=TOURNAMENT_SIZE, replace=False)

        # Find the index of the participant with the highest fitness
        best_participant_idx = participant_indices[np.argmax(fitnesses[participant_indices])]

        # Add the winning individual to the selected parents
        selected_parents.append(population[best_participant_idx])
    return selected_parents


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, swap_prob: float = 0.5, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each gene swaps between parents with probability swap_prob."""
    if parent_a.shape != parent_b.shape:
        raise ValueError("Parents must have the same shape.")

    rng = rng or np.random.default_rng()
    swap_mask = rng.random(parent_a.shape) < swap_prob

    offspring1 = np.where(swap_mask, parent_b, parent_a)
    offspring2 = np.where(swap_mask, parent_a, parent_b)
    return offspring1, offspring2


def mutate(genome: np.ndarray) -> np.ndarray:
    """Add gaussian noise to each gene with probability MUTATION_RATE. Returns a new array."""
    # Create a boolean mask for genes that will mutate
    mask = np.random.random(genome.shape) < MUTATION_RATE
    
    # Generate Gaussian noise with mean 0 and standard deviation MUTATION_STD
    noise = np.random.normal(loc=0.0, scale=MUTATION_SIGMA, size=genome.shape)
    
    # Apply noise only where the mask is True
    # Returns a new array automatically
    return np.where(mask, genome + noise, genome)


def evolve(alpha: float) -> dict:
    """
    Run the GA for N_GENERATIONS and log everything the analysis needs.

    Each generation:
        1. evaluate every genome WITH learning (alpha) -> fitness, learned genome
        2. evaluate every genome WITHOUT learning (alpha=0) -> innate fitness
        3. log, then build the next population (selection -> crossover -> mutation, we 
        optionally keep the best 1-2 unchanged = elitism)

    :param alpha: learning rate during life. Run once with ALPHA and once with 0 for GA-only.
    :return: log dict of numpy arrays, indexed by generation:
        'genomes' (n_gen, pop, GENOME_SIZE) starting weights
        'learned' (n_gen, pop, GENOME_SIZE) weights at end of life
        'fitness' (n_gen, pop) lifetime reward
        'innate_fitness' (n_gen, pop) reward with no learning
    """
    pass
