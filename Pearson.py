import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 在绘图时设置字体为支持Arial，并加粗
plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体为Arial
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题
plt.rcParams['font.weight'] = 'bold'  # 设置字体加粗
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['xtick.labelsize'] = 16  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16  # y轴刻度字体大小
plt.rcParams['axes.titlesize'] = 18   # 标题字体大小
plt.rcParams['mathtext.default'] = 'regular'  # 禁用LaTeX，取消斜体

# 示例数据
total_excel_file = 'data_v2_1.xlsx'  # 请确保文件路径正确

# 读取总集Excel文件
total_data = pd.read_excel(total_excel_file)
total_data = total_data.iloc[1:, :]  # 去掉表头后的数据

# 提取前12列的特征
X = total_data.iloc[:, :12]  # 前11列是自变量

# 计算皮尔森相关系数矩阵
corr_matrix = X.corr(method='pearson')

# 自定义颜色映射方案：从RGB(106, 142, 201)到RGB(232, 68, 70)
from matplotlib.colors import LinearSegmentedColormap

# 定义从RGB(106, 142, 201)到RGB(232, 68, 70)的颜色映射
colors = [(106/255, 142/255, 201/255), (239/255, 124/255, 127/255)]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# 绘制热力图，限制范围在 0.7 到 1.0
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt='.2f', linewidths=0.5, vmin=-0.7, vmax=1.0,
            xticklabels=['S$_x$', 'S$_y$', 'MP', 'DA$_{LO}$', 'DA$_{LB}$', 'D$_{LL}$',
                         'D$_{OO}$', 'RA$_{HB_1H}$', 'RA$_{HB_2H}$', 'DA$_{OB}$', 'MEB'],
            yticklabels=['S$_x$', 'S$_y$', 'MP', 'DA$_{LO}$', 'DA$_{LB}$', 'D$_{LL}$',
                         'D$_{OO}$', 'RA$_{HB_1H}$', 'RA$_{HB_2H}$', 'DA$_{OB}$', 'MEB'])

plt.title('Pearson Correlation Coefficient Heatmap', fontsize=20, weight='bold', pad=20)  # 标题字体加大

plt.show()
