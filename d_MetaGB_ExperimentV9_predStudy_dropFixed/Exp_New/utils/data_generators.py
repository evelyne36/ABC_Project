"""
Data Generators Module
Generates synthetic datasets for Experiment 2 (nonlinear and interaction pattern tests).
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


# =============================================================================
# Experiment 2A: Nonlinear Pattern Robustness Tests
# =============================================================================

def make_regression_square_noise(
    n_samples: int = 4000,
    n_features: int = 10,
    noise_std: float = 0.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate regression data with squared nonlinearity.

    y = sum(w_i * x_i^2) + noise

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        metadata: Dictionary with dataset description
    """
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    np.random.seed(random_state)

    # Generate features from standard normal distribution
    X = np.random.randn(n_samples, n_features)

    # Generate random weights
    weights = np.random.randn(n_features)

    # Compute target: y = sum(w_i * x_i^2)
    y = np.sum(weights * (X ** 2), axis=1)

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(n_samples) * noise_std
        y = y + noise

    metadata = {
        'type': 'square',
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_std': noise_std,
        'random_state': random_state,
        'formula': 'y = sum(w_i * x_i^2) + noise'
    }

    return X, y, metadata


def make_regression_sin_noise(
    n_samples: int = 4000,
    n_features: int = 10,
    noise_std: float = 0.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate regression data with sinusoidal nonlinearity.

    y = sum(w_i * sin(x_i)) + noise

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        metadata: Dictionary with dataset description
    """
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    np.random.seed(random_state)

    # Generate features from standard normal distribution
    X = np.random.randn(n_samples, n_features)

    # Generate random weights
    weights = np.random.randn(n_features)

    # Compute target: y = sum(w_i * sin(x_i))
    y = np.sum(weights * np.sin(X), axis=1)

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(n_samples) * noise_std
        y = y + noise

    metadata = {
        'type': 'sin',
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_std': noise_std,
        'random_state': random_state,
        'formula': 'y = sum(w_i * sin(x_i)) + noise'
    }

    return X, y, metadata


def make_regression_square_sin_noise(
    n_samples: int = 4000,
    n_features: int = 10,
    noise_std: float = 0.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate regression data with mixed squared and sinusoidal nonlinearity.

    y = sum(w_i * x_i^2) + sum(v_i * sin(x_i)) + noise

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        metadata: Dictionary with dataset description
    """
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    np.random.seed(random_state)

    # Generate features from standard normal distribution
    X = np.random.randn(n_samples, n_features)

    # Generate random weights for both components
    weights_square = np.random.randn(n_features)
    weights_sin = np.random.randn(n_features)

    # Compute target: y = sum(w_i * x_i^2) + sum(v_i * sin(x_i))
    y = np.sum(weights_square * (X ** 2), axis=1) + np.sum(weights_sin * np.sin(X), axis=1)

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(n_samples) * noise_std
        y = y + noise

    metadata = {
        'type': 'square_sin',
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_std': noise_std,
        'random_state': random_state,
        'formula': 'y = sum(w_i * x_i^2) + sum(v_i * sin(x_i)) + noise'
    }

    return X, y, metadata


def make_regression_sigmoid_noise(
    n_samples: int = 4000,
    n_features: int = 10,
    noise_std: float = 0.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate regression data with sigmoid nonlinearity.

    y = sum(w_i * sigmoid(x_i)) + noise
    where sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        metadata: Dictionary with dataset description
    """
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    np.random.seed(random_state)

    # Generate features from standard normal distribution
    X = np.random.randn(n_samples, n_features)

    # Generate random weights
    weights = np.random.randn(n_features)

    # Sigmoid function
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Compute target: y = sum(w_i * sigmoid(x_i))
    y = np.sum(weights * sigmoid(X), axis=1)

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(n_samples) * noise_std
        y = y + noise

    metadata = {
        'type': 'sigmoid',
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_std': noise_std,
        'random_state': random_state,
        'formula': 'y = sum(w_i * sigmoid(x_i)) + noise'
    }

    return X, y, metadata


# =============================================================================
# Experiment 2B: Feature Interaction Pattern Tests
# =============================================================================

def make_piecewise_linear(
    n_samples: int = 4000,
    n_features: int = 10,
    n_regions: int = 3,
    noise_std: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate piecewise linear regression data.

    The feature space is divided into regions based on x1 and x2.
    Each region has a different linear relationship with x3.

    Design:
        region = assign_region(x1, x2)  # Based on feature thresholds
        y = slope[region] * x3 + intercept[region] + noise

    Args:
        n_samples: Number of samples
        n_features: Number of features (minimum 3)
        n_regions: Number of piecewise regions (2-5)
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        metadata: Dictionary with dataset description
    """
    if n_features < 3:
        raise ValueError("n_features must be at least 3")
    if n_regions < 2 or n_regions > 5:
        raise ValueError("n_regions must be between 2 and 5")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    np.random.seed(random_state)

    # Generate features from standard normal distribution
    X = np.random.randn(n_samples, n_features)

    # Generate random slopes and intercepts for each region
    slopes = np.random.randn(n_regions) * 2  # Scale for variety
    intercepts = np.random.randn(n_regions) * 3

    # Assign regions based on x1 and x2
    # Create thresholds to divide the space
    if n_regions == 2:
        # Simple binary split based on x1
        regions = (X[:, 0] > 0).astype(int)
    elif n_regions == 3:
        # Three regions based on x1
        regions = np.zeros(n_samples, dtype=int)
        regions[X[:, 0] > 0.5] = 1
        regions[X[:, 0] > 1.0] = 2
    elif n_regions == 4:
        # Four regions based on x1 and x2 quadrants
        regions = np.zeros(n_samples, dtype=int)
        regions[(X[:, 0] > 0) & (X[:, 1] > 0)] = 0
        regions[(X[:, 0] > 0) & (X[:, 1] <= 0)] = 1
        regions[(X[:, 0] <= 0) & (X[:, 1] > 0)] = 2
        regions[(X[:, 0] <= 0) & (X[:, 1] <= 0)] = 3
    else:  # n_regions == 5
        # Five regions: center + 4 quadrants
        regions = np.zeros(n_samples, dtype=int)
        center_mask = (np.abs(X[:, 0]) < 0.5) & (np.abs(X[:, 1]) < 0.5)
        regions[center_mask] = 0
        regions[~center_mask & (X[:, 0] > 0) & (X[:, 1] > 0)] = 1
        regions[~center_mask & (X[:, 0] > 0) & (X[:, 1] <= 0)] = 2
        regions[~center_mask & (X[:, 0] <= 0) & (X[:, 1] > 0)] = 3
        regions[~center_mask & (X[:, 0] <= 0) & (X[:, 1] <= 0)] = 4

    # Compute target: y = slope[region] * x3 + intercept[region]
    y = slopes[regions] * X[:, 2] + intercepts[regions]

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(n_samples) * noise_std
        y = y + noise

    metadata = {
        'type': 'piecewise_linear',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_regions': n_regions,
        'noise_std': noise_std,
        'random_state': random_state,
        'slopes': slopes.tolist(),
        'intercepts': intercepts.tolist(),
        'formula': 'y = slope[region(x1,x2)] * x3 + intercept[region] + noise'
    }

    return X, y, metadata


def make_conditional_prediction(
    n_samples: int = 4000,
    n_features: int = 10,
    n_categories: int = 7,
    noise_std: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate conditional-prediction separation data.

    A categorical feature defines the "condition" (which prediction function to use).
    Continuous features define the "prediction" (specific value within the function).

    Design:
        category = categorical_feature  # e.g., day of week (0-6)
        y = slope[category] * continuous_feature + intercept[category] + noise

    Args:
        n_samples: Number of samples
        n_features: Number of features (1 categorical + rest continuous)
        n_categories: Number of categories (recommended 5-10)
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_features) - first column is categorical
        y: Target variable (n_samples,)
        metadata: Dictionary with dataset description
    """
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if n_categories < 2:
        raise ValueError("n_categories must be at least 2")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    np.random.seed(random_state)

    # Generate categorical feature (first column)
    categories = np.random.randint(0, n_categories, size=n_samples)

    # Generate continuous features
    continuous_features = np.random.randn(n_samples, n_features - 1)

    # Combine into feature matrix
    X = np.column_stack([categories, continuous_features])

    # Generate random slopes and intercepts for each category
    slopes = np.random.randn(n_categories) * 2
    intercepts = np.random.randn(n_categories) * 3

    # Compute target: y = slope[category] * x1 + intercept[category]
    # Use the first continuous feature (second column) as the main predictor
    y = slopes[categories] * X[:, 1] + intercepts[categories]

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(n_samples) * noise_std
        y = y + noise

    metadata = {
        'type': 'conditional_prediction',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_categories': n_categories,
        'noise_std': noise_std,
        'random_state': random_state,
        'slopes': slopes.tolist(),
        'intercepts': intercepts.tolist(),
        'categorical_feature_idx': 0,
        'formula': 'y = slope[category] * x_continuous + intercept[category] + noise'
    }

    return X, y, metadata


def make_sparse_interaction(
    n_samples: int = 4000,
    n_total_features: int = 100,
    n_relevant: int = 5,
    interaction_order: int = 2,
    noise_std: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate high-dimensional sparse interaction data.

    Only a few features are truly relevant, and they interact multiplicatively.

    Design (for interaction_order=2, n_relevant=5):
        y = x1*x2 + x3*x4 + x5 + noise
        (remaining features are pure noise)

    Design (for interaction_order=3, n_relevant=6):
        y = x1*x2*x3 + x4*x5*x6 + noise

    Args:
        n_samples: Number of samples
        n_total_features: Total number of features (including noise features)
        n_relevant: Number of relevant features
        interaction_order: Order of interaction (2 or 3)
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_total_features)
        y: Target variable (n_samples,)
        metadata: Dictionary with dataset description including relevant_features
    """
    if n_total_features < n_relevant:
        raise ValueError("n_total_features must be >= n_relevant")
    if n_relevant < interaction_order:
        raise ValueError("n_relevant must be >= interaction_order")
    if interaction_order not in [2, 3]:
        raise ValueError("interaction_order must be 2 or 3")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    np.random.seed(random_state)

    # Generate all features from standard normal distribution
    X = np.random.randn(n_samples, n_total_features)

    # Select relevant feature indices (first n_relevant features)
    relevant_features = list(range(n_relevant))

    # Compute target based on interaction order
    y = np.zeros(n_samples)

    if interaction_order == 2:
        # Create pairwise interactions
        n_pairs = n_relevant // 2
        for i in range(n_pairs):
            idx1, idx2 = 2 * i, 2 * i + 1
            y += X[:, idx1] * X[:, idx2]
        # If odd number of relevant features, add the last one linearly
        if n_relevant % 2 == 1:
            y += X[:, n_relevant - 1]
    else:  # interaction_order == 3
        # Create triplet interactions
        n_triplets = n_relevant // 3
        for i in range(n_triplets):
            idx1, idx2, idx3 = 3 * i, 3 * i + 1, 3 * i + 2
            y += X[:, idx1] * X[:, idx2] * X[:, idx3]
        # Handle remaining features
        remaining = n_relevant % 3
        if remaining == 2:
            y += X[:, n_relevant - 2] * X[:, n_relevant - 1]
        elif remaining == 1:
            y += X[:, n_relevant - 1]

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(n_samples) * noise_std
        y = y + noise

    # Build formula string
    if interaction_order == 2:
        terms = []
        n_pairs = n_relevant // 2
        for i in range(n_pairs):
            terms.append(f"x{2*i}*x{2*i+1}")
        if n_relevant % 2 == 1:
            terms.append(f"x{n_relevant-1}")
        formula = "y = " + " + ".join(terms) + " + noise"
    else:
        terms = []
        n_triplets = n_relevant // 3
        for i in range(n_triplets):
            terms.append(f"x{3*i}*x{3*i+1}*x{3*i+2}")
        remaining = n_relevant % 3
        if remaining == 2:
            terms.append(f"x{n_relevant-2}*x{n_relevant-1}")
        elif remaining == 1:
            terms.append(f"x{n_relevant-1}")
        formula = "y = " + " + ".join(terms) + " + noise"

    metadata = {
        'type': 'sparse_interaction',
        'n_samples': n_samples,
        'n_total_features': n_total_features,
        'n_relevant': n_relevant,
        'interaction_order': interaction_order,
        'noise_std': noise_std,
        'random_state': random_state,
        'relevant_features': relevant_features,
        'formula': formula
    }

    return X, y, metadata


# =============================================================================
# Utility Functions
# =============================================================================

def get_nonlinear_generators():
    """
    Get all nonlinear data generators for Experiment 2A.

    Returns:
        dict: Mapping from generator name to generator function
    """
    return {
        'square': make_regression_square_noise,
        'sin': make_regression_sin_noise,
        'square_sin': make_regression_square_sin_noise,
        'sigmoid': make_regression_sigmoid_noise
    }


def get_interaction_generators():
    """
    Get all interaction pattern generators for Experiment 2B.

    Returns:
        dict: Mapping from generator name to generator function
    """
    return {
        'piecewise_linear': make_piecewise_linear,
        'conditional_prediction': make_conditional_prediction,
        'sparse_interaction': make_sparse_interaction
    }


def generate_exp2a_datasets(noise_levels: List[float], random_state: int = 42):
    """
    Generate all datasets for Experiment 2A.

    Args:
        noise_levels: List of noise standard deviations to test
        random_state: Random seed

    Returns:
        list: List of (name, X, y, metadata) tuples
    """
    datasets = []
    generators = get_nonlinear_generators()

    for gen_name, gen_func in generators.items():
        for noise_std in noise_levels:
            X, y, metadata = gen_func(
                n_samples=4000,
                n_features=10,
                noise_std=noise_std,
                random_state=random_state
            )
            name = f"{gen_name}_noise{noise_std}"
            datasets.append((name, X, y, metadata))

    return datasets


def generate_exp2b_datasets(random_state: int = 42):
    """
    Generate all datasets for Experiment 2B.

    Args:
        random_state: Random seed

    Returns:
        list: List of (name, X, y, metadata) tuples
    """
    datasets = []

    # Piecewise linear variants
    for n_regions in [2, 3, 5]:
        X, y, metadata = make_piecewise_linear(
            n_samples=4000,
            n_features=10,
            n_regions=n_regions,
            noise_std=0.1,
            random_state=random_state
        )
        name = f"piecewise_{n_regions}regions"
        datasets.append((name, X, y, metadata))

    # Conditional prediction variants
    for n_categories in [5, 7, 10]:
        X, y, metadata = make_conditional_prediction(
            n_samples=4000,
            n_features=10,
            n_categories=n_categories,
            noise_std=0.1,
            random_state=random_state
        )
        name = f"conditional_{n_categories}cat"
        datasets.append((name, X, y, metadata))

    # Sparse interaction variants
    sparse_configs = [
        (20, 5, 2),   # 20 features, 5 relevant, 2-order interaction,3-order is too hard for models to learn, so we only test 2-order interaction in this experiment
        (20, 10, 2),   # 20 features, 10 relevant, 2-order interaction
        (50, 5, 2),   # 50 features, 5 relevant, 2-order interaction, 100 features is too slow.
        (50, 10, 2),   # 50 features, 10 relevant, 2-order interaction

    ]
    for n_total, n_rel, order in sparse_configs:
        X, y, metadata = make_sparse_interaction(
            n_samples=4000,
            n_total_features=n_total,
            n_relevant=n_rel,
            interaction_order=order,
            noise_std=0.1,
            random_state=random_state
        )
        name = f"sparse_{n_total}f_{n_rel}r_{order}order"
        datasets.append((name, X, y, metadata))

    return datasets


if __name__ == "__main__":
    # Test data generators
    print("Testing data generators...")

    # Test nonlinear generators
    print("\n=== Nonlinear Generators (Exp 2A) ===")
    for name, gen_func in get_nonlinear_generators().items():
        X, y, meta = gen_func(n_samples=100, noise_std=0.1)
        print(f"{name}: X.shape={X.shape}, y.shape={y.shape}")
        print(f"  Formula: {meta['formula']}")

    # Test interaction generators
    print("\n=== Interaction Generators (Exp 2B) ===")
    X, y, meta = make_piecewise_linear(n_samples=100, n_regions=3)
    print(f"piecewise_linear: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  Formula: {meta['formula']}")

    X, y, meta = make_conditional_prediction(n_samples=100, n_categories=7)
    print(f"conditional_prediction: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  Formula: {meta['formula']}")

    X, y, meta = make_sparse_interaction(n_samples=100, n_total_features=50, n_relevant=5)
    print(f"sparse_interaction: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  Formula: {meta['formula']}")
    print(f"  Relevant features: {meta['relevant_features']}")

    print("\nAll tests passed!")
