from CFRalgorithm import TempleCFR
import os

CHECKPOINT_DIR = 'checkpoints'
LATEST_CHECKPOINT = 'latest_checkpoint.txt'

def get_latest_checkpoint():
    if not os.path.exists(LATEST_CHECKPOINT):
        return None
    with open(LATEST_CHECKPOINT, 'r') as f:
        return f.read().strip()

def save_latest_checkpoint(checkpoint_path):
    with open(LATEST_CHECKPOINT, 'w') as f:
        f.write(checkpoint_path)

if __name__ == "__main__":
    latest_checkpoint_path = get_latest_checkpoint()

    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from {latest_checkpoint_path}")
        k = TempleCFR.load_checkpoint(latest_checkpoint_path)
    else:
        print("Starting new training session")
        k = TempleCFR(1_000_000, {}, {})

    try:
        utilities = k.cfr_iterations_external()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        k.save_checkpoint()
        save_latest_checkpoint(os.path.join(CHECKPOINT_DIR, f'cfr_checkpoint_{k.iteration}.pkl'))
        print("Checkpoint saved. Exiting.")

