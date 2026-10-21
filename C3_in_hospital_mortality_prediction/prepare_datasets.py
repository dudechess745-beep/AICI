# Prepare feature arrays and labels for this cohort.
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aici.data import main
if __name__ == '__main__':
    main(['prepare', '--cohort', 'C3'] + sys.argv[1:])
