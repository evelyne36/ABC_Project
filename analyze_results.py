"""
结果分析和可视化脚本
从实验结果JSON文件中读取数据并生成可视化图表
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 设置图表样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class ResultAnalyzer:
    """结果分析器类"""
    
    def __init__(self, results_dir='results', plots_dir='plots'):
        """
        初始化结果分析器
        
        参数:
            results_dir: 结果目录
            plots_dir: 图表保存目录
        """
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
    
    def load_results(self, json_file):
        """
        加载JSON结果文件
        
        参数:
            json_file: JSON文件路径
            
        返回:
            结果字典
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_module_A(self, results):
        """
        分析模块A结果（参数理解实验）
        
        参数:
            results: 模块A结果字典
        """
        print("\n" + "="*80)
        print("分析模块A: 参数理解实验")
        print("="*80)
        
        metagb_results = results['metagb']
        xgb_results = results['xgboost']
        
        # 找到最佳模型
        best_metagb = min(metagb_results, key=lambda x: x['val_mse_mean'])
        best_xgb = min(xgb_results, key=lambda x: x['val_mse_mean'])
        
        print("\n最佳 MetaGB 模型:")
        print(f"  参数: {best_metagb['params']}")
        print(f"  训练集 MSE: {best_metagb['train_mse_mean']:.4f} ± {best_metagb['train_mse_std']:.4f}")
        print(f"  验证集 MSE: {best_metagb['val_mse_mean']:.4f} ± {best_metagb['val_mse_std']:.4f}")
        
        print("\n最佳 XGBoost 模型:")
        print(f"  参数: {best_xgb['params']}")
        print(f"  训练集 MSE: {best_xgb['train_mse_mean']:.4f} ± {best_xgb['train_mse_std']:.4f}")
        print(f"  验证集 MSE: {best_xgb['val_mse_mean']:.4f} ± {best_xgb['val_mse_std']:.4f}")
        
        # 参数敏感性分析
        self.plot_parameter_sensitivity_metagb(metagb_results)
        self.plot_parameter_sensitivity_xgb(xgb_results)
        
        # 模型对比
        self.plot_model_comparison(best_metagb, best_xgb)
    
    def plot_parameter_sensitivity_metagb(self, results):
        """绘制MetaGB参数敏感性分析"""
        params_list = ['n_estimators', 'max_depth', 'learning_rate', 
                     'pred_feature_fraction', 'subsample']
        
        for param in params_list:
            param_values = sorted(set([r['params'][param] for r in results]))
            avg_val_mse = []
            std_val_mse = []
            
            for value in param_values:
                relevant = [r for r in results if r['params'][param] == value]
                avg_val_mse.append(np.mean([r['val_mse_mean'] for r in relevant]))
                std_val_mse.append(np.mean([r['val_mse_std'] for r in relevant]))
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(param_values, avg_val_mse, yerr=std_val_mse, 
                        fmt='o-', linewidth=2, markersize=8, capsize=5)
            plt.xlabel(param, fontsize=12)
            plt.ylabel('验证集 MSE', fontsize=12)
            plt.title(f'MetaGB: {param} 参数敏感性', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, 
                                  f'module_A_metagb_{param}_sensitivity.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"图表已保存: {save_path}")
    
    def plot_parameter_sensitivity_xgb(self, results):
        """绘制XGBoost参数敏感性分析"""
        params_list = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        
        for param in params_list:
            param_values = sorted(set([r['params'][param] for r in results]))
            avg_val_mse = []
            std_val_mse = []
            
            for value in param_values:
                relevant = [r for r in results if r['params'][param] == value]
                avg_val_mse.append(np.mean([r['val_mse_mean'] for r in relevant]))
                std_val_mse.append(np.mean([r['val_mse_std'] for r in relevant]))
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(param_values, avg_val_mse, yerr=std_val_mse, 
                        fmt='s-', linewidth=2, markersize=8, capsize=5, color='red')
            plt.xlabel(param, fontsize=12)
            plt.ylabel('验证集 MSE', fontsize=12)
            plt.title(f'XGBoost: {param} 参数敏感性', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, 
                                  f'module_A_xgb_{param}_sensitivity.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"图表已保存: {save_path}")
    
    def plot_model_comparison(self, best_metagb, best_xgb):
        """绘制模型对比图"""
        models = ['MetaGB', 'XGBoost']
        train_mse = [best_metagb['train_mse_mean'], best_xgb['train_mse_mean']]
        val_mse = [best_metagb['val_mse_mean'], best_xgb['val_mse_mean']]
        train_std = [best_metagb['train_mse_std'], best_xgb['train_mse_std']]
        val_std = [best_metagb['val_mse_std'], best_xgb['val_mse_std']]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, train_mse, width, yerr=train_std, 
                      label='训练集 MSE', alpha=0.8, capsize=5)
        rects2 = ax.bar(x + width/2, val_mse, width, yerr=val_std, 
                      label='验证集 MSE', alpha=0.8, capsize=5)
        
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title('最佳模型对比（糖尿病数据集）', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)
        
        add_labels(rects1)
        add_labels(rects2)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'module_A_model_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {save_path}")
    
    def analyze_module_B(self, results):
        """
        分析模块B结果（合成数据集测试）
        
        参数:
            results: 模块B结果字典
        """
        print("\n" + "="*80)
        print("分析模块B: 合成数据集性能测试")
        print("="*80)
        
        data_types = ['square', 'sin', 'square_sin', 'sigmoid']
        noise_levels = sorted(set([r['noise_level'] for r in results['results']]))
        
        # 绘制噪声鲁棒性曲线
        for pred_frac in [0.0, 0.4, 0.8]:
            self.plot_noise_robustness_by_pred_fraction(
                results, pred_frac, data_types, noise_levels
            )
        
        # 绘制不同数据类型的性能对比
        self.plot_data_type_comparison(results, data_types, noise_levels)
        
        # 绘制pred_feature_fraction影响
        self.plot_pred_fraction_impact(results, data_types)
    
    def plot_noise_robustness_by_pred_fraction(self, results, pred_frac, 
                                            data_types, noise_levels):
        """绘制不同pred_fraction下的噪声鲁棒性"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, data_type in enumerate(data_types):
            ax = axes[idx]
            
            # 收集该数据类型的结果
            data_type_results = [r for r in results['results'] 
                               if r['data_type'] == data_type]
            
            # MetaGB不同pred_fraction
            pred_key = f'pred_{pred_frac}'
            metagb_rmse = [r['metagb'][pred_key]['test_rmse'] 
                          for r in data_type_results]
            
            # XGBoost
            xgb_rmse = [r['xgboost']['test_rmse'] for r in data_type_results]
            
            ax.plot(noise_levels, metagb_rmse, 'o-', linewidth=2, 
                   markersize=8, label=f'MetaGB (pred_frac={pred_frac})')
            ax.plot(noise_levels, xgb_rmse, 's--', linewidth=2, 
                   markersize=8, label='XGBoost')
            
            ax.set_xlabel('噪声水平', fontsize=10)
            ax.set_ylabel('测试集 RMSE', fontsize=10)
            ax.set_title(f'{data_type} 数据', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'噪声鲁棒性分析 (pred_feature_fraction={pred_frac})', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, 
                              f'module_B_noise_robustness_pred{pred_frac}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {save_path}")
    
    def plot_data_type_comparison(self, results, data_types, noise_levels):
        """绘制不同数据类型的性能对比（无噪声）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 无噪声情况
        noise_0_results = [r for r in results['results'] if r['noise_level'] == 0.0]
        
        metagb_results = {}
        for pred_frac in [0.0, 0.4, 0.8]:
            metagb_results[pred_frac] = []
            for r in noise_0_results:
                metagb_results[pred_frac].append(
                    r['metagb'][f'pred_{pred_frac}']['test_rmse']
                )
        
        xgb_results = [r['xgboost']['test_rmse'] for r in noise_0_results]
        
        # MetaGB对比
        ax = axes[0]
        x = np.arange(len(data_types))
        width = 0.25
        
        ax.bar(x - width, metagb_results[0.0], width, label='MetaGB pred=0.0', alpha=0.8)
        ax.bar(x, metagb_results[0.4], width, label='MetaGB pred=0.4', alpha=0.8)
        ax.bar(x + width, metagb_results[0.8], width, label='MetaGB pred=0.8', alpha=0.8)
        ax.set_ylabel('测试集 RMSE', fontsize=12)
        ax.set_title('MetaGB: 不同数据类型性能对比（无噪声）', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(data_types)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # XGBoost vs 最佳MetaGB
        ax = axes[1]
        ax.bar(x - width/2, metagb_results[0.4], width, label='MetaGB (best)', alpha=0.8)
        ax.bar(x + width/2, xgb_results, width, label='XGBoost', alpha=0.8)
        ax.set_ylabel('测试集 RMSE', fontsize=12)
        ax.set_title('最佳模型对比（无噪声）', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(data_types)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'module_B_data_type_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {save_path}")
    
    def plot_pred_fraction_impact(self, results, data_types):
        """绘制pred_feature_fraction的影响"""
        pred_fracs = [0.0, 0.2, 0.4, 0.6, 0.8]
        
        for data_type in data_types:
            # 收集该数据类型无噪声的结果
            data_type_results = [r for r in results['results'] 
                              if r['data_type'] == data_type and r['noise_level'] == 0.0]
            
            if not data_type_results:
                continue
            
            result = data_type_results[0]
            
            metagb_rmse = [result['metagb'][f'pred_{pf}']['test_rmse'] 
                          for pf in pred_fracs]
            xgb_rmse = result['xgboost']['test_rmse']
            xgb_rmse_list = [xgb_rmse] * len(pred_fracs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(pred_fracs, metagb_rmse, 'o-', linewidth=2, 
                   markersize=8, label='MetaGB')
            plt.plot(pred_fracs, xgb_rmse_list, 's--', linewidth=2, 
                   markersize=8, label='XGBoost (baseline)')
            plt.xlabel('pred_feature_fraction', fontsize=12)
            plt.ylabel('测试集 RMSE', fontsize=12)
            plt.title(f'pred_feature_fraction 影响 - {data_type} 数据', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, 
                                  f'module_B_{data_type}_pred_fraction.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"图表已保存: {save_path}")
    
    def analyze_module_C(self, results):
        """
        分析模块C结果（TabArena数据集测试）
        
        参数:
            results: 模块C结果字典
        """
        print("\n" + "="*80)
        print("分析模块C: TabArena数据集批量测试")
        print("="*80)
        
        datasets = results['datasets']
        
        # 收集数据
        dataset_names = []
        xgb_test_rmse = []
        metagb_best_rmse = []
        metagb_best_pred_frac = []
        
        for dataset_name, dataset_result in datasets.items():
            dataset_names.append(dataset_name)
            xgb_test_rmse.append(dataset_result['xgboost']['test_rmse'])
            
            # 找到最佳MetaGB配置
            metagb_results = dataset_result['metagb']
            best_metagb = min(metagb_results, key=lambda x: x['test_rmse'])
            metagb_best_rmse.append(best_metagb['test_rmse'])
            metagb_best_pred_frac.append(best_metagb['pred_feature_fraction'])
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Dataset': dataset_names,
            'XGBoost_RMSE': xgb_test_rmse,
            'MetaGB_RMSE': metagb_best_rmse,
            'MetaGB_Best_Pred_Frac': metagb_best_pred_frac
        })
        
        # 计算改进率
        df['Improvement_Rate'] = (df['XGBoost_RMSE'] - df['MetaGB_RMSE']) / df['XGBoost_RMSE'] * 100
        
        # 按改进率排序
        df = df.sort_values('Improvement_Rate', ascending=False)
        
        print("\n" + "-"*80)
        print("TabArena数据集性能排名:")
        print("-"*80)
        print(df.to_string(index=False))
        print("-"*80)
        
        # 绘制性能对比
        self.plot_tabarena_performance(df)
        
        # 绘制改进率分布
        self.plot_improvement_distribution(df)
        
        # 绘制最佳pred_fraction统计
        self.plot_best_pred_fraction_dist(df)
    
    def plot_tabarena_performance(self, df):
        """绘制TabArena数据集性能对比"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['XGBoost_RMSE'], width, 
                      label='XGBoost', alpha=0.8)
        bars2 = ax.bar(x + width/2, df['MetaGB_RMSE'], width, 
                      label='MetaGB (best)', alpha=0.8)
        
        ax.set_xlabel('数据集', fontsize=12)
        ax.set_ylabel('测试集 RMSE', fontsize=12)
        ax.set_title('TabArena数据集: MetaGB vs XGBoost 性能对比', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Dataset'], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加改进率标注
        for i, (xgb_rmse, metagb_rmse, imp_rate) in enumerate(
            zip(df['XGBoost_RMSE'], df['MetaGB_RMSE'], df['Improvement_Rate'])):
            
            if imp_rate > 0:
                color = 'green'
            else:
                color = 'red'
            
            ax.annotate(f'{imp_rate:.1f}%', 
                      xy=(i + width/2, metagb_rmse),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom', 
                      fontsize=8, color=color, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'module_C_tabarena_performance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {save_path}")
    
    def plot_improvement_distribution(self, df):
        """绘制改进率分布"""
        plt.figure(figsize=(10, 6))
        
        # 直方图
        plt.hist(df['Improvement_Rate'], bins=10, edgecolor='black', alpha=0.7)
        
        # 添加均值线
        mean_improvement = df['Improvement_Rate'].mean()
        plt.axvline(mean_improvement, color='red', linestyle='--', linewidth=2, 
                   label=f'平均改进率: {mean_improvement:.2f}%')
        
        # 添加零线
        plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        plt.xlabel('改进率 (%)', fontsize=12)
        plt.ylabel('数据集数量', fontsize=12)
        plt.title('MetaGB相对XGBoost的改进率分布', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'module_C_improvement_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {save_path}")
    
    def plot_best_pred_fraction_dist(self, df):
        """绘制最佳pred_fraction分布"""
        pred_frac_counts = df['MetaGB_Best_Pred_Frac'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        pred_frac_counts.plot(kind='bar', color='steelblue', alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for i, count in enumerate(pred_frac_counts):
            plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('最佳 pred_feature_fraction', fontsize=12)
        plt.ylabel('数据集数量', fontsize=12)
        plt.title('最佳 pred_feature_fraction 分布', fontsize=14)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'module_C_best_pred_fraction.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {save_path}")
    
    def generate_summary_report(self, results):
        """
        生成汇总报告
        
        参数:
            results: 完整结果字典
        """
        print("\n" + "="*80)
        print("生成汇总报告")
        print("="*80)
        
        report_path = os.path.join(self.results_dir, 'summary_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MetaGB 实验汇总报告\n\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
            
            # 模块A总结
            if 'module_A' in results:
                f.write("## 模块A: 参数理解实验（糖尿病数据集）\n\n")
                metagb_results = results['module_A']['metagb']
                best_metagb = min(metagb_results, key=lambda x: x['val_mse_mean'])
                f.write(f"- 最佳MetaGB验证集MSE: {best_metagb['val_mse_mean']:.4f}\n")
                f.write(f"- 最佳参数: {best_metagb['params']}\n\n")
            
            # 模块B总结
            if 'module_B' in results:
                f.write("## 模块B: 合成数据集性能测试\n\n")
                f.write(f"- 测试数据类型数量: 4\n")
                f.write(f"- 噪声水平数量: 6\n")
                f.write("- 总测试组合数: 24\n\n")
            
            # 模块C总结
            if 'module_C' in results:
                f.write("## 模块C: TabArena数据集批量测试\n\n")
                datasets = results['module_C']['datasets']
                f.write(f"- 测试数据集数量: {len(datasets)}\n")
                
                # 统计改进情况
                improvements = []
                for dataset_result in datasets.values():
                    xgb_rmse = dataset_result['xgboost']['test_rmse']
                    metagb_results = dataset_result['metagb']
                    best_metagb = min(metagb_results, key=lambda x: x['test_rmse'])
                    metagb_rmse = best_metagb['test_rmse']
                    imp_rate = (xgb_rmse - metagb_rmse) / xgb_rmse * 100
                    improvements.append(imp_rate)
                
                better_count = sum(1 for imp in improvements if imp > 0)
                avg_improvement = np.mean(improvements)
                
                f.write(f"- MetaGB优于XGBoost的数据集数: {better_count}/{len(datasets)}\n")
                f.write(f"- 平均改进率: {avg_improvement:.2f}%\n\n")
        
        print(f"汇总报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("结果分析和可视化")
    print("="*80)
    
    analyzer = ResultAnalyzer()
    
    # 查找最新的结果文件
    results_files = glob.glob(os.path.join('results', '**', '*.json'), recursive=True)
    
    if not results_files:
        print("未找到结果文件!")
        return
    
    print(f"\n找到 {len(results_files)} 个结果文件")
    
    # 分析模块A
    module_a_files = [f for f in results_files if 'module_A' in f]
    if module_a_files:
        print(f"\n分析模块A结果 ({len(module_a_files)} 个文件)...")
        for file in module_a_files:
            try:
                results = analyzer.load_results(file)
                analyzer.analyze_module_A(results)
            except Exception as e:
                print(f"  错误: {e}")
    
    # 分析模块B
    module_b_files = [f for f in results_files if 'module_B' in f]
    if module_b_files:
        print(f"\n分析模块B结果 ({len(module_b_files)} 个文件)...")
        for file in module_b_files:
            try:
                results = analyzer.load_results(file)
                analyzer.analyze_module_B(results)
            except Exception as e:
                print(f"  错误: {e}")
    
    # 分析模块C
    module_c_files = [f for f in results_files if 'module_C' in f]
    if module_c_files:
        print(f"\n分析模块C结果 ({len(module_c_files)} 个文件)...")
        for file in module_c_files:
            try:
                results = analyzer.load_results(file)
                analyzer.analyze_module_C(results)
            except Exception as e:
                print(f"  错误: {e}")
    
    # 生成汇总报告
    all_results = {}
    for results_type in ['A', 'B', 'C']:
        module_files = [f for f in results_files if f'module_{results_type}' in f]
        if module_files:
            try:
                results = analyzer.load_results(module_files[0])
                all_results[f'module_{results_type}'] = results
            except:
                pass
    
    if all_results:
        analyzer.generate_summary_report(all_results)
    
    print("\n" + "="*80)
    print("分析和可视化完成!")
    print(f"图表保存在: {analyzer.plots_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
