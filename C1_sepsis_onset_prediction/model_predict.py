# Load trained student models and predict outcomes for this cohort.
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aici.pipeline import main
if __name__ == '__main__':
    main(['predict'] + sys.argv[1:])
