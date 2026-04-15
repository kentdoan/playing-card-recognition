import numpy as np
import os
from helpers import extract_cards

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

TRAIN_DIR = os.path.join(_PROJECT_ROOT, 'dataset')
TEMPLATE_FILE = os.path.join(_SCRIPT_DIR, 'card_templates.npz')

SUITS = ['diamonds', 'clubs', 'hearts', 'spades']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

def train_system():
    templates = {}
    success_count = 0

    for suit in SUITS:
        for rank in RANKS:
            card_name = f"{rank}{suit}"
            img_path = os.path.join(TRAIN_DIR, f"{card_name}.jpg")

            if not os.path.exists(img_path):
                print(f"Warning: missing training sample {img_path}")
                continue

            cards, _ = extract_cards(img_path, is_training=True)
            if len(cards) == 1:
                templates[card_name] = cards[0]['warped']
                success_count += 1
            else:
                print(f"Error: unable to extract one clear card from {img_path}")

    np.savez_compressed(TEMPLATE_FILE, **templates)
    print(f"Training finished: saved {success_count}/52 templates to {TEMPLATE_FILE}")


if __name__ == '__main__':
    train_system()
