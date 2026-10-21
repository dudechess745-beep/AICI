# Define XGBoost teacher hyperparameter candidates and run teacher training.
SEARCH_SPACE = {'max_depth': [3, 4], 'learning_rate': [0.002, 0.005, 0.02], 'subsample': [0.4, 0.45, 0.5, 0.55, 0.6], 'colsample_bytree': [0.4, 0.45, 0.5, 0.55, 0.6], 'reg_alpha': [0.0, 0.001, 0.01, 0.02], 'reg_lambda': [0.05, 0.1, 0.5, 1]}
if __name__ == '__main__':
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from aici.pipeline import main
    main(['fit', '--stage', 'teacher', '--config', str(ROOT / 'configs/C2.json')] + sys.argv[1:])
