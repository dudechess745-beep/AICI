# Define XGBoost teacher hyperparameter candidates and run teacher training.
SEARCH_SPACE = {'max_depth': [3, 4], 'learning_rate': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9], 'reg_alpha': [0.0, 0.005, 0.01, 0.05, 0.1], 'reg_lambda': [0.8, 1, 1.5, 2, 4]}
if __name__ == '__main__':
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from aici.pipeline import main
    main(['fit', '--stage', 'teacher', '--config', str(ROOT / 'configs/C3.json')] + sys.argv[1:])
