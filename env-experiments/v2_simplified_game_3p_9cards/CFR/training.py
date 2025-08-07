from CFRalgorithm import TempleCFR
import os
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser(description='CFR Training Script')
parser.add_argument('--iterations', type=int, default=1_000_000, help='Number of CFR iterations to run')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--truthful_attackers', type=bool, default = False, help='Whether attackers are truthful')
args = parser.parse_args()  

CHECKPOINT_DIR = args.checkpoint_dir
LATEST_CHECKPOINT = 'latest_checkpoint.txt'

def get_latest_checkpoint():
    if not os.path.exists(LATEST_CHECKPOINT):
        return None
    with open(LATEST_CHECKPOINT, 'r') as f:
        return f.read().strip()

def save_latest_checkpoint(checkpoint_path):
    with open(LATEST_CHECKPOINT, 'w') as f:
        f.write(checkpoint_path)

# Set argv where attacker truthful flag is passed either True or False is received with anrg name attacker-truthful


if __name__ == "__main__":
    latest_checkpoint_path = get_latest_checkpoint()

    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from {latest_checkpoint_path}")
        k = TempleCFR.load_checkpoint(latest_checkpoint_path)
    else:
        print("Starting new training session")
        k = TempleCFR(args.iterations, {}, {})
        print(args)
    print("Attackers are truthful:", args.truthful_attackers)
    try:
        utilities = k.cfr_iterations_external(args.truthful_attackers)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        k.save_checkpoint()
        save_latest_checkpoint(os.path.join(CHECKPOINT_DIR, f'cfr_checkpoint_{k.iteration}.pkl'))
        print("Checkpoint saved. Exiting.")

