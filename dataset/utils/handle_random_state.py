
import random


def handle_random_state(random_state: int = None) -> None:
    if random_state is not None:
        random.seed(random_state)
