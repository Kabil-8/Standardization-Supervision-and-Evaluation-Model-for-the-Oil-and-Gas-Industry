for pkg in ['torch', 'sklearn', 'lightgbm', 'xgboost', 'catboost', 'joblib']:
    try:
        m = __import__(pkg)
        print(f'{pkg}: {getattr(m, "__version__", "installed")}')
    except ImportError as e:
        print(f'{pkg}: NOT INSTALLED ({e})')
