"""
快速测试脚本 - 验证实验环境配置
"""

import sys
import os

print("="*80)
print("MetaGB 实验环境 - 快速测试")
print("="*80)

# 测试1: 检查Python版本
print("\n[测试1] Python版本")
print(f"  Python版本: {sys.version}")
major, minor = sys.version_info[:2]
if major >= 3 and minor >= 8:
    print("  ✓ Python版本符合要求 (>=3.8)")
else:
    print("  ✗ Python版本过低，建议使用3.8以上")

# 测试2: 检查依赖包
print("\n[测试2] 检查依赖包")
required_packages = {
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-learn',
    'xgboost': 'XGBoost',
    'cffi': 'CFFI'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (未安装)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n  请安装缺失的包:")
    print(f"  pip install {' '.join(missing_packages)}")

# 测试3: 检查Matplotlib和Seaborn
print("\n[测试3] 检查可视化库")
viz_packages = {'matplotlib': 'Matplotlib', 'seaborn': 'Seaborn'}
for package, name in viz_packages.items():
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (未安装，可选，用于结果可视化)")

# 测试4: 检查MetaGB编译库
print("\n[测试4] 检查MetaGB编译库")
metagb_found = False
if os.path.exists('libmetagb.dll'):
    print("  ✓ 找到 libmetagb.dll (Windows)")
    metagb_found = True
elif os.path.exists('libmetagb.so'):
    print("  ✓ 找到 libmetagb.so (Linux/Mac)")
    metagb_found = True
elif os.path.exists('ABC_Project/metagb/libmetagb.dll'):
    print("  ✓ 找到 libmetagb.dll (在metagb目录)")
    metagb_found = True
elif os.path.exists('ABC_Project/metagb/libmetagb.so'):
    print("  ✓ 找到 libmetagb.so (在metagb目录)")
    metagb_found = True
else:
    print("  ✗ 未找到MetaGB编译库")
    print("     请运行: python ABC_Project/metagb/build_metagb.py")

# 测试5: 检查数据集
print("\n[测试5] 检查数据集")
dataset_dir = 'dataset/tabArena_regression'
if os.path.exists(dataset_dir):
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    print(f"  ✓ 找到 {len(csv_files)} 个TabArena数据集")
    if len(csv_files) < 12:
        print(f"    注意: 预期12个数据集，当前{len(csv_files)}个")
    for csv_file in csv_files[:3]:
        print(f"    - {csv_file}")
    if len(csv_files) > 3:
        print(f"    ... (还有{len(csv_files)-3}个)")
else:
    print(f"  ✗ 数据集目录不存在: {dataset_dir}")

# 测试6: 测试MetaGB导入
if metagb_found:
    print("\n[测试6] 测试MetaGB导入")
    try:
        sys.path.insert(0, os.path.join('ABC_Project', 'metagb'))
        from metagb_utils import MetaGB
        print("  ✓ MetaGB导入成功")
        
        # 创建一个简单的测试模型
        import numpy as np
        X = np.random.randn(100, 5).astype(np.float64)
        y = np.random.randn(100).astype(np.float64)
        
        model = MetaGB(
            learning_rate=0.1,
            num_trees=10,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            min_data_in_leaf=10,
            max_depth=3,
            max_bin=32,
            version=3,
            seed=42
        )
        
        print("  ✓ MetaGB模型创建成功")
        model.fit(X, y)
        print("  ✓ MetaGB模型训练成功")
        predictions = model.predict(X)
        print(f"  ✓ MetaGB预测成功 (输出形状: {predictions.shape})")
        
    except Exception as e:
        print(f"  ✗ MetaGB测试失败: {e}")
else:
    print("\n[测试6] 测试MetaGB导入 (跳过 - 未找到编译库)")

# 测试7: 检查输出目录
print("\n[测试7] 检查输出目录")
for dir_name in ['results', 'logs', 'plots']:
    if os.path.exists(dir_name):
        print(f"  ✓ {dir_name}/ 目录存在")
    else:
        print(f"  ○ {dir_name}/ 目录不存在 (实验时将自动创建)")

# 总结
print("\n" + "="*80)
print("测试总结")
print("="*80)

all_tests_passed = (
    len(missing_packages) == 0 and
    metagb_found
)

if all_tests_passed:
    print("✓ 环境配置完整，可以开始实验！")
    print("\n快速开始:")
    print("  - 运行所有实验: python run_experiments.py --module all")
    print("  - 或者双击运行: run_all.bat")
else:
    print("✗ 环境配置不完整，请解决上述问题后再运行实验")
    print("\n参考文档:")
    print("  - 详细说明: README_EXPERIMENTS.md")
    print("  - 常见问题: 见README_EXPERIMENTS.md的'常见问题'部分")

print("="*80)
