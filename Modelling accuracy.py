import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['mathtext.default'] = 'regular' 

def train_lgbm_regression(total_excel_path):

    total_data = pd.read_excel(total_excel_path)
    total_data = total_data.iloc[1:, :] 


    X = total_data.iloc[:, :12].values 
    y = total_data.iloc[:, 12].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150 / 600)

    model = LGBMRegressor(n_estimators=150)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_path = 'lgbm_model.joblib'
    joblib.dump(model, model_path)
    print(f"模型已保存为 {model_path}")

    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_pred, color=(106 / 255, 142 / 255, 201 / 255), label='Measured value', alpha=0.8)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--',
             color=(232 / 255, 68 / 255, 70 / 255), label='True value')

    plt.xlabel('DFT Calculated', fontsize=18, fontweight='bold', labelpad=10)
    plt.ylabel('ML Predicted', fontsize=18, fontweight='bold', labelpad=10)
    plt.title('LGBM Regression Model Performance', fontsize=20, fontweight='bold', pad=15)

    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.legend(loc='lower right', fontsize=14, bbox_to_anchor=(1, 0))

    plt.text(0.05, 0.96, f'$R^2$: {r2:.6f}', transform=plt.gca().transAxes, fontsize=14, fontweight='bold',verticalalignment='top')
    plt.text(0.05, 0.87, f'MSE: {mse:.6f}', transform=plt.gca().transAxes, fontsize=14, fontweight='bold', verticalalignment='top')
    plt.text(0.05, 0.80, f'MAE: {mae:.6f}', transform=plt.gca().transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

    plt.tight_layout()
    plt.show()

    return mse, mae, r2

total_excel_file = 'data_v2_1.xlsx' 
mse, mae, r2 = train_lgbm_regression(total_excel_file)
print(f"测试集上的均方误差 (MSE): {mse}")
print(f"测试集上的平均绝对误差 (MAE): {mae}")
print(f"测试集上的决定系数 ($R^2$): {r2}")
