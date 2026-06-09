#!/usr/bin/env python3
"""
Entry point for the modular track curator (v6).

Usage:
    python run_curator.py [folder] [--csv ...] [--mask ...] [--image ...] [--auto-thresholds]

This is a thin wrapper; all logic lives in the `curator` package. The
QApplication is bootstrapped inside curator.app at import time, before any Qt
widget can be constructed.
"""

from curator.app import main

if __name__ == "__main__":
    main()
