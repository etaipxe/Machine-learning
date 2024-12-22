import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap 
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap

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

X = total_data.iloc[:, :10].values
y = total_data.iloc[:, 10].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150/600)

model = LGBMRegressor(n_estimators=150)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

colors = [(106/255, 142/255, 201/255), (239/255, 124/255, 127/255)]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

feature_names = ['S$_x$', 'S$_y$', 'MP', 'DA$_{LO}$', 'DA$_{LB}$', 'D$_{LL}$',
                 'D$_{OO}$', 'RA$_{HB_1H}$', 'RA$_{HB_2H}$', 'DA$_{OB}$']

shap.summary_plot(shap_values, X_train, feature_names=feature_names, cmap=cmap, show=False)

fig = plt.gcf()

if len(fig.axes) > 1:
    colorbar = fig.axes[-1]
    colorbar.tick_params(labelsize=14)

plt.gcf().axes[0].set_title('SHAP value (impact on model output)', fontsize=20, weight='bold', pad=10)
plt.gcf().axes[0].set_xlabel('')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()
