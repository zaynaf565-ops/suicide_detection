"""Entry point to run classical model training from the project root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_classical import main

if __name__ == "__main__":
    main()
