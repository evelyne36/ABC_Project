"""
Test script to verify metagb can be imported correctly
"""
import sys
import os

# Add metagb to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metagb'))

try:
    # Test importing libmetagb (compiled C library)
    import libmetagb
    print("✓ libmetagb imported successfully")

    # Test importing metagb_utils (Python wrapper)
    from metagb_utils import MetaGB, MetaGBRegressor
    print("✓ metagb_utils.MetaGB imported successfully")
    print("✓ metagb_utils.MetaGBRegressor imported successfully")

    # Test basic functionality
    print("\nTesting basic MetaGB instantiation...")
    model = MetaGB(
        learning_rate=0.1,
        num_trees=10,
        max_depth=3,
        version=3,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        min_data_in_leaf=10,
        max_bin=255,
    )
    print("✓ MetaGB model created successfully")

    # Test MetaGBRegressor alias
    model2 = MetaGBRegressor(
        learning_rate=0.1,
        num_trees=10,
        max_depth=3,
        version=3,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        min_data_in_leaf=10,
        max_bin=255,
    )
    print("✓ MetaGBRegressor (alias) model created successfully")

    print("\n✅ All imports working correctly!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease compile metagb first:")
    print("  cd metagb")
    print("  python build_metagb.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
