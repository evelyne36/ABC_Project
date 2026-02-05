"""
数据加载模块
统一加载数据集的标准接口
"""

import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_diabetes_dataset(test_size=0.2, val_size=0.2, random_state=42, scale_y=False):
    """
    加载糖尿病数据集并进行标准划分
    
    参数:
        test_size: 测试集比例
        val_size: 验证集比例（训练集的子集）
        random_state: 随机种子
        scale_y: 是否对目标值y进行标准化
    
    返回:
        X_train, X_val, X_test, y_train, y_val, y_test, (y_scaler)
        如果scale_y=True，额外返回用于反标准化的scaler对象
    """
    data = load_diabetes()
    X, y = data.data, data.target
    
    # 先划分训练集和测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 从训练集中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )
    
    # 对目标值进行标准化（如果需要）
    y_scaler = None
    if scale_y:
        # 使用训练集的统计信息进行标准化
        y_scaler = StandardScaler()
        y_train_reshaped = y_train.reshape(-1, 1)
        y_train = y_scaler.fit_transform(y_train_reshaped)
        y_train = y_train.reshape(-1)
        
        # 使用相同的scaler验证集和测试集
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    
    if scale_y:
        return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_housing_dataset(test_size=0.2, val_size=0.2, random_state=42, scale_y=False):
    """
    加载加州房价数据集并进行标准划分
    
    参数:
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        scale_y: 是否对目标值y进行标准化
    
    返回:
        X_train, X_val, X_test, y_train, y_val, y_test, (y_scaler)
        如果scale_y=True，额外返回用于反标准化的scaler对象
    """
    housing = fetch_openml(name="house_prices", as_frame=True)
    df = housing.frame
    
    # 处理缺失值
    df = df.select_dtypes(include=[np.number]).dropna()
    
    X = df.drop('target', axis=1).values if 'target' in df.columns else df.values
    # 假设最后一列是目标
    y = df.iloc[:, -1].values
    if X.shape[1] == y.shape[0]:
        X = df.iloc[:, :-1].values
    
    # 划分数据
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )
    
    # 对目标值进行标准化（如果需要）
    y_scaler = None
    if scale_y:
        # 使用训练集的统计信息进行标准化
        y_scaler = StandardScaler()
        y_train_reshaped = y_train.reshape(-1, 1)
        y_train = y_scaler.fit_transform(y_train_reshaped)
        y_train = y_train.reshape(-1)
        
        # 使用相同的scaler验证集和测试集
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    
    if scale_y:
        return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_csv_dataset(csv_path, target_col=None, test_size=0.2, val_size=0.2, random_state=42, scale_y=False):
    """
    加载CSV格式数据集
    
    参数:
        csv_path: CSV文件路径
        target_col: 目标列名/索引（None表示最后一列）
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        scale_y: 是否对目标值y进行标准化
    
    返回:
        X_train, X_val, X_test, y_train, y_val, y_test, (y_scaler)
        如果scale_y=True，额外返回用于反标准化的scaler对象
    """
    df = pd.read_csv(csv_path)
    
    # 处理缺失值
    df = df.select_dtypes(include=[np.number]).dropna()
    
    if target_col is None:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    elif isinstance(target_col, str):
        X = df.drop(target_col, axis=1).values
        y = df[target_col].values
    else:
        X = df.iloc[:, :target_col].values
        X = np.concatenate([df.iloc[:, target_col+1:].values], axis=1)
        y = df.iloc[:, target_col].values
    
    # 划分数据
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )
    
    # 对目标值进行标准化（如果需要）
    y_scaler = None
    if scale_y:
        # 使用训练集的统计信息进行标准化
        y_scaler = StandardScaler()
        y_train_reshaped = y_train.reshape(-1, 1)
        y_train = y_scaler.fit_transform(y_train_reshaped)
        y_train = y_train.reshape(-1)
        
        # 使用相同的scaler验证集和测试集
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    
    if scale_y:
        return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test


def get_tabarena_dataset_list():
    """
    获取TabArena回归数据集的CSV文件列表
    
    返回:
        数据集文件名列表
    """
    data_dir = "dataset/tabArena_regression"
    datasets = [
        "airfoil_self_noise.csv",
        "Another-Dataset-on-used-Fiat-500.csv",
        "concrete_compressive_strength.csv",
        "diamonds.csv",
        "Food_Delivery_Time.csv",
        "healthcare_insurance_expenses.csv",
        "houses.csv",
        "miami_housing.csv",
        "physiochemical_protein.csv",
        "QSAR_fish_toxicity.csv",
        "QSAR-TID-11.csv",
        "superconductivity.csv",
        "wine_quality.csv"
    ]
    return [os.path.join(data_dir, ds) for ds in datasets]


def create_kfold_splits(X, y, n_splits=5, random_state=42):
    """
    创建K折交叉验证的数据划分
    
    参数:
        X: 特征数据
        y: 目标数据
        n_splits: 折数
        random_state: 随机种子
    
    返回:
        包含(train_idx, val_idx)的生成器
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return kfold.split(X, y)
