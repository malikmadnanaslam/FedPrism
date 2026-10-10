"""Run the leakage-controlled SAPEx-D evaluation and create result artifacts.

The runner saves held-out predictions, bootstrap uncertainty, paired tests,
ablations, privacy-accounting details, and experiment figures under
``results/``. Dataset file: ``data/sapex_d.csv``.

Run: python main.py
"""
from experiment import run


if __name__ == "__main__":
    run()
