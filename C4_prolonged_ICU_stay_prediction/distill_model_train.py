# Run student distillation for this cohort using the trained teacher.
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aici.pipeline import main
if __name__ == '__main__':
    main(['fit', '--stage', 'student', '--config', str(ROOT / 'configs/C4.json')] + sys.argv[1:])
