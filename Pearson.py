import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['mathtext.default'] = 'regular'

total_excel_file = 'data_v2_1.xlsx'

total_data = pd.read_excel(total_excel_file)
total_data = total_data.iloc[1:, :]

X = total_data.iloc[:, :12]

corr_matrix = X.corr(method='pearson')

from matplotlib.colors import LinearSegmentedColormap

colors = [(106/255, 142/255, 201/255), (239/255, 124/255, 127/255)]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt='.2f', linewidths=0.5, vmin=-0.7, vmax=1.0,
            xticklabels=['S$_x$', 'S$_y$', 'MP', 'DA$_{LO}$', 'DA$_{LB}$', 'D$_{LL}$',
                         'D$_{OO}$', 'RA$_{HB_1H}$', 'RA$_{HB_2H}$', 'DA$_{OB}$', 'MEB'],
            yticklabels=['S$_x$', 'S$_y$', 'MP', 'DA$_{LO}$', 'DA$_{LB}$', 'D$_{LL}$',
                         'D$_{OO}$', 'RA$_{HB_1H}$', 'RA$_{HB_2H}$', 'DA$_{OB}$', 'MEB'])

plt.title('Pearson Correlation Coefficient Heatmap', fontsize=20, weight='bold', pad=20)

plt.show()
