"""
Test Script for Exp_New
Verifies that all modules can be imported and basic functionality works.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ABC_Project', 'metagb'))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from utils.hyperparams import (
            RANDOM_SEED, TRAIN_TEST_SPLIT, CV_FOLDS,
            get_xgb_param_grid, get_metagb_param_grid,
            get_fixed_xgb_params, get_fixed_metagb_params,
            get_model_variants_exp2, NOISE_LEVELS
        )
        print("  [OK] hyperparams")
    except Exception as e:
        print(f"  [FAIL] hyperparams: {e}")
        return False

    try:
        from utils.data_generators import (
            make_regression_square_noise,
            make_regression_sin_noise,
            make_regression_square_sin_noise,
            make_regression_sigmoid_noise,
            make_piecewise_linear,
            make_conditional_prediction,
            make_sparse_interaction,
            get_nonlinear_generators,
            generate_exp2b_datasets
        )
        print("  [OK] data_generators")
    except Exception as e:
        print(f"  [FAIL] data_generators: {e}")
        return False

    try:
        # Import directly from trainer module to avoid __init__.py issues
        import utils.trainer as trainer_module
        _ = trainer_module.create_xgboost_model
        _ = trainer_module.create_metagb_model
        _ = trainer_module.train_single_model
        _ = trainer_module.evaluate_model
        _ = trainer_module.train_with_grid_search
        _ = trainer_module.calculate_metrics
        print("  [OK] trainer")
    except Exception as e:
        print(f"  [FAIL] trainer: {e}")
        return False

    try:
        from utils.logger import (
            setup_experiment_logger,
            ExperimentLogger
        )
        print("  [OK] logger")
    except Exception as e:
        print(f"  [FAIL] logger: {e}")
        return False

    try:
        from utils.visualizer import (
            plot_training_curve,
            plot_noise_robustness,
            plot_training_curves_comparison,
            plot_tree_depth_comparison
        )
        print("  [OK] visualizer")
    except Exception as e:
        print(f"  [FAIL] visualizer: {e}")
        return False

    return True


def test_data_generators():
    """Test data generator functions."""
    print("\nTesting data generators...")

    import numpy as np
    from utils.data_generators import (
        make_regression_square_noise,
        make_regression_sin_noise,
        make_piecewise_linear,
        make_conditional_prediction,
        make_sparse_interaction
    )

    tests = [
        ("square_noise", lambda: make_regression_square_noise(n_samples=100, noise_std=0.1)),
        ("sin_noise", lambda: make_regression_sin_noise(n_samples=100, noise_std=0.1)),
        ("piecewise_linear", lambda: make_piecewise_linear(n_samples=100, n_regions=3)),
        ("conditional_prediction", lambda: make_conditional_prediction(n_samples=100, n_categories=5)),
        ("sparse_interaction", lambda: make_sparse_interaction(n_samples=100, n_total_features=50, n_relevant=5)),
    ]

    for name, gen_func in tests:
        try:
            X, y, meta = gen_func()
            assert X.shape[0] == 100, f"Expected 100 samples, got {X.shape[0]}"
            assert y.shape[0] == 100, f"Expected 100 targets, got {y.shape[0]}"
            assert 'type' in meta, "Missing 'type' in metadata"
            print(f"  [OK] {name}: X={X.shape}, y={y.shape}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            return False

    return True


def test_model_creation():
    """Test model creation functions."""
    print("\nTesting model creation...")

    # Import directly from trainer module to avoid __init__.py issues
    import utils.trainer as trainer_module
    create_xgboost_model = trainer_module.create_xgboost_model
    create_metagb_model = trainer_module.create_metagb_model
    from utils.hyperparams import get_fixed_xgb_params, get_fixed_metagb_params

    # Test XGBoost creation
    try:
        xgb_params = get_fixed_xgb_params()
        xgb_model = create_xgboost_model(xgb_params)
        print(f"  [OK] XGBoost model created")
    except Exception as e:
        print(f"  [FAIL] XGBoost creation: {e}")
        return False

    # Test MetaGB creation
    try:
        metagb_params = get_fixed_metagb_params(pred_feature_fraction=0.4)
        metagb_model = create_metagb_model(metagb_params)
        print(f"  [OK] MetaGB model created")
    except Exception as e:
        print(f"  [FAIL] MetaGB creation: {e}")
        return False

    return True


def test_small_training():
    """Test training on small synthetic data."""
    print("\nTesting small training run...")

    import numpy as np
    from utils.data_generators import make_regression_square_noise
    # Import directly from trainer module
    import utils.trainer as trainer_module
    train_single_model = trainer_module.train_single_model
    evaluate_model = trainer_module.evaluate_model
    from utils.hyperparams import get_fixed_xgb_params, get_fixed_metagb_params
    from sklearn.model_selection import train_test_split

    # Generate small dataset
    X, y, _ = make_regression_square_noise(n_samples=200, noise_std=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Test XGBoost training
    try:
        xgb_params = get_fixed_xgb_params()
        xgb_params['n_estimators'] = 10  # Small for testing

        result = train_single_model(
            model_type='xgboost',
            params=xgb_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            record_loss_curve=False
        )

        test_eval = evaluate_model(result['model'], 'xgboost', X_test, y_test)
        print(f"  [OK] XGBoost training: RMSE={test_eval['test_metrics']['rmse']:.4f}")
    except Exception as e:
        print(f"  [FAIL] XGBoost training: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test MetaGB training
    try:
        metagb_params = get_fixed_metagb_params(pred_feature_fraction=0.4)
        metagb_params['n_estimators'] = 10  # Small for testing

        result = train_single_model(
            model_type='metagb',
            params=metagb_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            record_loss_curve=False
        )

        test_eval = evaluate_model(result['model'], 'metagb', X_test, y_test)
        print(f"  [OK] MetaGB training: RMSE={test_eval['test_metrics']['rmse']:.4f}")
    except Exception as e:
        print(f"  [FAIL] MetaGB training: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_dataset_loading():
    """Test loading TabArena datasets."""
    print("\nTesting dataset loading...")

    import pandas as pd
    import numpy as np

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'tabArena_regression')

    if not os.path.exists(data_dir):
        print(f"  [SKIP] Dataset directory not found: {data_dir}")
        return True

    datasets = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not datasets:
        print(f"  [SKIP] No CSV files found in {data_dir}")
        return True

    # Test loading first dataset
    test_file = os.path.join(data_dir, datasets[0])
    try:
        df = pd.read_csv(test_file)
        df = df.select_dtypes(include=[np.number]).dropna()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        print(f"  [OK] Loaded {datasets[0]}: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"  [FAIL] Loading {datasets[0]}: {e}")
        return False

    print(f"  Found {len(datasets)} datasets in total")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Exp_New Setup Verification")
    print("=" * 60)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test data generators
    if not test_data_generators():
        all_passed = False

    # Test model creation
    if not test_model_creation():
        all_passed = False

    # Test small training
    if not test_small_training():
        all_passed = False

    # Test dataset loading
    if not test_dataset_loading():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("The experiment setup is ready to run.")
    else:
        print("SOME TESTS FAILED!")
        print("Please check the errors above before running experiments.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
