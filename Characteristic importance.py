import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular' 

def train_lgbm_regression_with_importance_pie(total_excel_path):
   
    total_data = pd.read_excel(total_excel_path)

    total_data = total_data.iloc[1:, :]

    X = total_data.iloc[:, :10].values 
    y = total_data.iloc[:, 10].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150 / 600)

    model = LGBMRegressor(n_estimators=150)

    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    feature_names = total_data.columns[:10] 

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=True)

    sorted_features = importance_df['Feature'].values
    sorted_importances = importance_df['Importance'].values

    cmap = LinearSegmentedColormap.from_list("custom_gradient",
                                             [(106 / 255, 142 / 255, 201 / 255), (239 / 255, 124 / 255, 127 / 255)],
                                             N=len(sorted_features))
    colors = [cmap(i / len(sorted_features)) for i in range(len(sorted_features))]

    latex_labels = [
        'S$_x$', 'S$_y$', 'MP', 'DA$_{LO}$', 'DA$_{LB}$', 'D$_{LL}$',
        'D$_{OO}$', 'RA$_{HB_1H}$', 'RA$_{HB_2H}$', 'DA$_{OB}$'
    ]

    sorted_latex_labels = [latex_labels[np.where(feature_names == feature)[0][0]] for feature in sorted_features]

    plt.figure(figsize=(10, 10))
    wedges, texts, autotexts = plt.pie(
        sorted_importances,
        labels=sorted_latex_labels, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )

    for text in texts + autotexts:
        text.set_fontsize(14)
        text.set_fontweight('bold')

    plt.axis('equal') 

    plt.tight_layout()
    plt.show()

total_excel_file = 'data_v2_1.xlsx' 
train_lgbm_regression_with_importance_pie(total_excel_file)
