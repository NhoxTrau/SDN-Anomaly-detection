#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_PARENT = SCRIPT_DIR.parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from sdn_nids_realtime.train_v2.poll_sequence_builder import main

if __name__ == "__main__":
    main()
