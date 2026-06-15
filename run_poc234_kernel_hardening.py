import sys
from pathlib import Path


SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from highway.runners.run_poc234_kernel_hardening import main


if __name__ == "__main__":
    main()


