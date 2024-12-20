import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'  # 禁用LaTeX，取消斜体

def train_lgbm_regression_with_importance_pie(total_excel_path):
    """
    从总集Excel文件中随机划分训练集（450条）和测试集（150条），
    训练LGBM回归模型并绘制特征重要性的饼图（按照重要性由小到大排序）。
    :param total_excel_path: 总集Excel文件路径（共600条数据）
    :return: None
    """
    # 读取总集Excel文件
    total_data = pd.read_excel(total_excel_path)

    # 去掉表头后的数据
    total_data = total_data.iloc[1:, :]

    # 划分自变量X和因变量Y
    X = total_data.iloc[:, :10].values  # 前12列是自变量
    y = total_data.iloc[:, 10].values  # 第13列是因变量

    # 随机划分训练集（450条）和测试集（150条）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150 / 600)

    # 初始化LGBM回归模型
    model = LGBMRegressor(n_estimators=150)

    # 训练模型
    model.fit(X_train, y_train)

    # 获取特征重要性
    feature_importances = model.feature_importances_
    feature_names = total_data.columns[:10]  # 根据数据的特征列名

    # 将特征重要性和特征名一起排序，按照重要性从小到大排序
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=True)

    # 提取排序后的特征名和重要性
    sorted_features = importance_df['Feature'].values
    sorted_importances = importance_df['Importance'].values

    # 创建从RGB(106,142,201)到RGB(232,68,70)的渐变色方案
    cmap = LinearSegmentedColormap.from_list("custom_gradient",
                                             [(106 / 255, 142 / 255, 201 / 255), (239 / 255, 124 / 255, 127 / 255)],
                                             N=len(sorted_features))
    colors = [cmap(i / len(sorted_features)) for i in range(len(sorted_features))]

    # 使用 LaTeX 语法手动设置标签
    latex_labels = [
        'S$_x$', 'S$_y$', 'MP', 'DA$_{LO}$', 'DA$_{LB}$', 'D$_{LL}$',
        'D$_{OO}$', 'RA$_{HB_1H}$', 'RA$_{HB_2H}$', 'DA$_{OB}$'
    ]

    # 根据排序后的特征名提取对应的LaTeX标签
    sorted_latex_labels = [latex_labels[np.where(feature_names == feature)[0][0]] for feature in sorted_features]

    # 绘制特征重要性的饼图（按照重要性由小到大排序）
    plt.figure(figsize=(10, 10))
    wedges, texts, autotexts = plt.pie(
        sorted_importances,
        labels=sorted_latex_labels,  # 使用排序后的LaTeX标签
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )

    # 设置标签、数值的字体、大小和加粗样式
    for text in texts + autotexts:
        text.set_fontsize(14)
        text.set_fontweight('bold')

    plt.axis('equal')  # 确保饼图是圆形的

    # 调整图形与边缘的距离
    plt.tight_layout()
    plt.show()

# 示例用法
total_excel_file = 'data_v2_1.xlsx'  # 请确保文件路径正确
train_lgbm_regression_with_importance_pie(total_excel_file)
