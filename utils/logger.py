"""
日志记录模块
用于统一管理实验日志记录
"""

import logging
import os
from datetime import datetime


def setup_logger(exp_type, dataset_name, seed, model_name, log_dir="logs"):
    """
    设置实验日志记录器
    
    参数:
        exp_type: 实验类型 (A, B, C, E)
        dataset_name: 数据集名称
        seed: 随机种子
        model_name: 模型名称
        log_dir: 日志保存目录
    
    返回:
        logger对象
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{exp_type}-{dataset_name}-{seed}-{model_name}-{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置日志
    logger = logging.getLogger(f"{exp_type}_{dataset_name}_{seed}_{model_name}")
    logger.setLevel(logging.DEBUG)
    
    # 防止重复添加handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件保存至: {log_path}")
    return logger


def log_experiment_info(logger, exp_type, dataset, seed, model, params):
    """
    记录实验基本信息
    
    参数:
        logger: logger对象
        exp_type: 实验类型
        dataset: 数据集名称
        seed: 随机种子
        model: 模型名称
        params: 参数字典
    """
    logger.info("=" * 80)
    logger.info("实验信息")
    logger.info("=" * 80)
    logger.info(f"实验类型: {exp_type}")
    logger.info(f"数据集: {dataset}")
    logger.info(f"随机种子: {seed}")
    logger.info(f"模型: {model}")
    logger.info(f"参数配置:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 80)


def log_results(logger, results_dict):
    """
    记录实验结果
    
    参数:
        logger: logger对象
        results_dict: 结果字典
    """
    logger.info("=" * 80)
    logger.info("实验结果")
    logger.info("=" * 80)
    for k, v in results_dict.items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 80)


class ExperimentLogger:
    """
    简化的实验日志记录器类
    用于主实验脚本的日志记录
    """
    
    def __init__(self, log_file):
        """
        初始化日志记录器
        
        参数:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        self.logger = logging.getLogger('MainExperiment')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除现有handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 文件handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def log(self, message, level='INFO'):
        """
        记录日志
        
        参数:
            message: 日志消息
            level: 日志级别 (INFO, DEBUG, WARNING, ERROR)
        """
        if level == 'DEBUG':
            self.logger.debug(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)
        else:
            self.logger.info(message)
    
    def close(self):
        """关闭日志"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
