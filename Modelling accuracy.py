import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import numpy as np

# 设置字体和图形样式
plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体为Arial
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.weight'] = 'bold'  # 设置字体加粗
plt.rcParams['mathtext.default'] = 'regular'  # 禁用LaTeX，取消斜体

def train_lgbm_regression(total_excel_path):
    # 读取总集Excel文件
    total_data = pd.read_excel(total_excel_path)
    total_data = total_data.iloc[1:, :]  # 去掉表头后的数据

    # 划分自变量X和因变量Y
    X = total_data.iloc[:, :12].values  # 前12列是自变量
    y = total_data.iloc[:, 12].values  # 第13列是因变量

    # 随机划分训练集（450条）和测试集（150条）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150 / 600)

    # 初始化LGBM回归模型
    model = LGBMRegressor(n_estimators=150)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集结果
    y_pred = model.predict(X_test)

    # 计算模型评估指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 保存模型到文件
    model_path = 'lgbm_model.joblib'
    joblib.dump(model, model_path)
    print(f"模型已保存为 {model_path}")

    # 创建图形
    plt.figure(figsize=(8, 6))

    # 绘制真实值为x轴，预测值为y轴的散点图
    plt.scatter(y_test, y_pred, color=(106 / 255, 142 / 255, 201 / 255), label='Measured value', alpha=0.8)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--',
             color=(232 / 255, 68 / 255, 70 / 255), label='True value')

    # 设置标签和标题
    plt.xlabel('DFT Calculated', fontsize=18, fontweight='bold', labelpad=10)  # 设置x轴标题与标签间距
    plt.ylabel('ML Predicted', fontsize=18, fontweight='bold', labelpad=10)    # 设置y轴标题与标签间距
    plt.title('LGBM Regression Model Performance', fontsize=20, fontweight='bold', pad=15)  # 增加标题与图之间距离

    # 设置x、y轴的刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 调整标签位置
    plt.legend(loc='lower right', fontsize=14, bbox_to_anchor=(1, 0))

    # 显示MSE、MAE、R^2在图中
    plt.text(0.05, 0.96, f'$R^2$: {r2:.6f}', transform=plt.gca().transAxes, fontsize=14, fontweight='bold',verticalalignment='top')
    plt.text(0.05, 0.87, f'MSE: {mse:.6f}', transform=plt.gca().transAxes, fontsize=14, fontweight='bold', verticalalignment='top')
    plt.text(0.05, 0.80, f'MAE: {mae:.6f}', transform=plt.gca().transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

    # 显示图形
    plt.tight_layout()
    plt.show()

    return mse, mae, r2


# 示例用法
total_excel_file = 'data_v2_1.xlsx'  # 请确保文件路径正确
mse, mae, r2 = train_lgbm_regression(total_excel_file)
print(f"测试集上的均方误差 (MSE): {mse}")
print(f"测试集上的平均绝对误差 (MAE): {mae}")
print(f"测试集上的决定系数 ($R^2$): {r2}")
