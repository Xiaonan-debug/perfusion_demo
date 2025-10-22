import random
import init  # Assumes init.big_state exists
from operations import single_step

def run_simulation(steps=100):
    action_combo = [] * 2
    big_state = init.initial_big_state()  # Initial big_state
    print("")
    print("")
    print(big_state)

    for step in range(steps):
        action_value = random.randint(0, 19682)
        action_combo = [action_value, big_state]
        big_state, newScoreCard, scoreValue = single_step(action_combo)

        # Optional: print progress or state
        print(f"Step {step + 1}: action={action_value}, state={big_state}")

    
# Example usage
if __name__ == "__main__":
    final_state = run_simulation(steps=100)  # Or any number of steps
