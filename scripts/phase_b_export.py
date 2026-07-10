#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seeded_unet.phase_b_export import main  # noqa: E402

if __name__ == "__main__":
    main()
