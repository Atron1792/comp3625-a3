import numpy as np

from ga import evolve, ALPHA
import analysis


def main(seed: int = 0, out_path: str = "logs.npz"):
    """
    1. Seed random / numpy.
    2. log_learn = evolve(alpha=ALPHA)  # GA + learning
    3. log_ga_only = evolve(alpha=0.0)  # control: GA alone
    4. Save both logs to out_path (so plots can be redone without rerunning).
    5. Call the three plot functions in analysis.py.
    (Later, if time allows: repeat over several seeds and report mean +/- std.)
    """
    pass


if __name__ == "__main__":
    main()
