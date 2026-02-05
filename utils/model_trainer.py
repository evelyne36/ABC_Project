"""
模型训练模块
封装XGBoost和MetaGB的训练过程，支持损失曲线记录
"""

import sys
import os
import numpy as np
import time
from itertools import product

# 添加ABC_Project/metagb路径以导入MetaGB
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ABC_Project', 'metagb'))

from xgboost import XGBRegressor
from metagb_utils import MetaGB


class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self, model_type='xgboost', **model_params):
        """
        初始化模型训练器
        
        参数:
            model_type: 模型类型 ('xgboost', 'metagb_v0', 'metagb_v3')
            model_params: 模型参数
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.train_losses = []
        self.val_losses = []
    
    def create_model(self):
        """根据类型创建模型"""
        if self.model_type == 'xgboost':
            self.model = XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                n_jobs=1,
                **self.model_params
            )
        elif self.model_type in ['metagb_v0', 'metagb_v3']:
            version = 3 if self.model_type == 'metagb_v3' else 0
            self.model = MetaGB(
                learning_rate=self.model_params.get('learning_rate', 0.1),
                num_trees=self.model_params.get('n_estimators', 100),
                feature_fraction=self.model_params.get('colsample_bytree', 0.8),
                pred_feature_fraction=self.model_params.get('pred_feature_fraction', 
                                                           self.model_params.get('colsample_bytree', 0.8)),
                bagging_fraction=self.model_params.get('subsample', 1.0),
                min_data_in_leaf=self.model_params.get('min_data_in_leaf', 20),
                max_depth=self.model_params.get('max_depth', 5),
                max_bin=self.model_params.get('max_bin', 64),
                lambda_l2=self.model_params.get('lambda_l2', 0.0),
                version=version,
                seed=self.model_params.get('random_state', 42)
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train_with_loss_tracking(self, X_train, y_train, X_val, y_val, 
                                  eval_interval=1, logger=None):
        """
        训练模型并记录损失曲线
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            eval_interval: 每隔多少轮记录损失
            logger: 日志记录器
        
        返回:
            训练时间
        """
        self.create_model()
        self.train_losses = []
        self.val_losses = []
        
        start_time = time.time()
        
        num_trees = self.model_params.get('n_estimators', 100)
        
        if self.model_type == 'xgboost':
            # XGBoost使用eval_result回调记录损失
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
            for i in range(num_trees):
                self.model.fit(X_train, y_train, 
                              eval_set=eval_set,
                              verbose=False)
                
                # 每隔eval_interval轮记录一次
                if (i + 1) % eval_interval == 0 or (i + 1) == num_trees:
                    train_pred = self.model.predict(X_train)
                    val_pred = self.model.predict(X_val)
                    
                    train_mse = np.mean((y_train - train_pred) ** 2)
                    val_mse = np.mean((y_val - val_pred) ** 2)
                    
                    self.train_losses.append(train_mse)
                    self.val_losses.append(val_mse)
                    
                    if logger and (i + 1) % (eval_interval * 10) == 0:
                        logger.info(f"  轮次 {i+1}/{num_trees}: 训练MSE={train_mse:.4f}, 验证MSE={val_mse:.4f}")
        
        else:  # MetaGB
            # MetaGB每次训练前num_trees棵树，需要逐步训练
            learning_rate = self.model_params.get('learning_rate', 0.1)
            
            for i in range(num_trees):
                # MetaGB不支持增量训练，需要按指定树数重新训练
                self.model.num_trees = i + 1
                self.model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
                
                if (i + 1) % eval_interval == 0 or (i + 1) == num_trees:
                    train_pred = self.model.predict(X_train.astype(np.float64))
                    val_pred = self.model.predict(X_val.astype(np.float64))
                    
                    train_mse = np.mean((y_train - train_pred) ** 2)
                    val_mse = np.mean((y_val - val_pred) ** 2)
                    
                    self.train_losses.append(train_mse)
                    self.val_losses.append(val_mse)
                    
                    if logger and (i + 1) % (eval_interval * 10) == 0:
                        logger.info(f"  轮次 {i+1}/{num_trees}: 训练MSE={train_mse:.4f}, 验证MSE={val_mse:.4f}")
        
        train_time = time.time() - start_time
        
        if logger:
            logger.info(f"训练完成，耗时: {train_time:.2f}秒")
            logger.info(f"最终训练MSE: {self.train_losses[-1]:.4f}")
            logger.info(f"最终验证MSE: {self.val_losses[-1]:.4f}")
        
        return train_time
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 输入特征
        
        返回:
            预测结果
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用train_with_loss_tracking")
        
        if self.model_type == 'xgboost':
            return self.model.predict(X)
        else:  # MetaGB
            return self.model.predict(X.astype(np.float64))


def train_single_model(model_type, params, X_train, y_train, X_val, y_val, 
                        logger=None, eval_interval=1):
    """
    训练单个模型并返回结果
    
    参数:
        model_type: 模型类型
        params: 参数字典
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        logger: 日志记录器
        eval_interval: 每隔多少轮记录损失
    
    返回:
        包含结果和模型的字典
    """
    trainer = ModelTrainer(model_type=model_type, **params)
    
    train_time = trainer.train_with_loss_tracking(
        X_train, y_train, X_val, y_val, 
        eval_interval=eval_interval, 
        logger=logger
    )
    
    # 计算各项指标
    train_pred = trainer.predict(X_train)
    val_pred = trainer.predict(X_val)
    
    train_mse = np.mean((y_train - train_pred) ** 2)
    val_mse = np.mean((y_val - val_pred) ** 2)
    rmse = np.sqrt(val_mse)
    mae = np.mean(np.abs(y_val - val_pred))
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    ss_res = np.sum((y_val - val_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'model': trainer.model,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_time': train_time,
        'final_train_mse': trainer.train_losses[-1] if trainer.train_losses else train_mse,
        'final_val_mse': trainer.val_losses[-1] if trainer.val_losses else val_mse
    }


def grid_search(model_type, param_grid, X_train, y_train, X_val, y_val, 
                logger=None, eval_interval=5):
    """
    网格搜索
    
    参数:
        model_type: 模型类型
        param_grid: 参数网格
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        logger: 日志记录器
        eval_interval: 评估间隔
    
    返回:
        结果列表
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    total_combinations = np.prod([len(v) for v in param_values])
    
    if logger:
        logger.info(f"\n开始网格搜索: {model_type}")
        logger.info(f"参数组合数: {total_combinations}")
        logger.info("=" * 80)
    
    results = []
    
    for i, params in enumerate(product(*param_values)):
        param_dict = dict(zip(param_names, params))
        
        if logger:
            logger.info(f"\n[{i+1}/{total_combinations}] 参数组合: {param_dict}")
        
        try:
            result = train_single_model(
                model_type, param_dict, X_train, y_train, X_val, y_val,
                logger=logger,
                eval_interval=eval_interval
            )
            result['params'] = param_dict
            results.append(result)
            
            if logger:
                logger.info(f"  训练MSE: {result['train_mse']:.4f} | 验证MSE: {result['val_mse']:.4f} | 时间: {result['train_time']:.2f}s")
        
        except Exception as e:
            if logger:
                logger.error(f"  训练失败: {str(e)}")
            continue
    
    return results
