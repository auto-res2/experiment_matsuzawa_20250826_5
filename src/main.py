"""
main.py – project entry point (`python -m src.main`).  The CLI mirrors
that of the paper: choose which experiment to run or execute all.
"""
from __future__ import annotations

import argparse

from .evaluate import quick_test, run_exp1, run_exp2, run_exp3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        choices=["1", "2", "3", "all", "test"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    args = parser.parse_args()

    if args.exp == "1":
        run_exp1()
    elif args.exp == "2":
        run_exp2()
    elif args.exp == "3":
        run_exp3()
    elif args.exp == "test":
        quick_test()
    else:  # all
        run_exp1(); run_exp2(); run_exp3()


if __name__ == "__main__":
    main()
