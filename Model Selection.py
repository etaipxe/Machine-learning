import pandas as pd
from sklearn.linear_model import (LinearRegression, Ridge, HuberRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Set font weight
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'

def train_and_compare_models(total_excel_path, param_grids, n_runs=25):
    total_data = pd.read_excel(total_excel_path).iloc[1:, :]
    X = total_data.iloc[:, :12].values
    y = total_data.iloc[:, 12].values

    models = {
        "LR": LinearRegression(),
        "RR": Ridge(),
        "HR": HuberRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "RF": RandomForestRegressor(),
        "GB": GradientBoostingRegressor(),
        "ET": ExtraTreesRegressor(),
        "AB": AdaBoostRegressor(),
        "LGB": LGBMRegressor(),
        "XGB": XGBRegressor(),
        "CB": CatBoostRegressor(),
    }

    rmse_scores = []
    r2_scores = []
    model_names = []
    best_params_list = []  # Store best parameters

    for name, model in models.items():
        rmse_list = []
        r2_list = []
        best_params = None

        if name in param_grids:
            print(f"Performing GridSearchCV for {name}...")

        for _ in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150 / 600)

            # Perform GridSearchCV if parameters are available
            if name in param_grids:
                grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            rmse_list.append(rmse)
            r2_list.append(r2)

        avg_rmse = np.mean(rmse_list)
        avg_r2 = np.mean(r2_list)

        rmse_scores.append(avg_rmse)
        r2_scores.append(avg_r2)
        model_names.append(name)
        best_params_list.append(best_params if best_params else "N/A")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Upper plot: RMSE comparison
    bars1 = ax1.bar(model_names, rmse_scores, color=(106 / 255, 142 / 255, 201 / 255))
    ax1.set_ylabel('RMSE (eV)', fontsize=16, fontweight='bold')
    ax1.set_xticklabels(model_names, fontsize=12, fontweight='bold', rotation=45)

    # Set RMSE y-ticks
    rmse_ticks = np.arange(0.02, max(rmse_scores) + 0.01, 0.02)
    ax1.set_yticks(rmse_ticks)

    # Highlight the lowest RMSE bar
    min_rmse_idx = np.argmin(rmse_scores)
    bars1[min_rmse_idx].set_color((232 / 255, 68 / 255, 70 / 255))

    # Lower plot: R² comparison
    bars2 = ax2.bar(model_names, r2_scores, color=(106 / 255, 142 / 255, 201 / 255))
    ax2.set_ylabel('$R^2$', fontsize=16, fontweight='bold')
    ax2.set_xticklabels(model_names, fontsize=12, fontweight='bold', rotation=45)

    # Set R² y-ticks
    r2_ticks = np.arange(0, 1.0, 0.2)
    ax2.set_yticks(r2_ticks)

    # Highlight the highest R² bar
    max_r2_idx = np.argmax(r2_scores)
    bars2[max_r2_idx].set_color((232 / 255, 68 / 255, 70 / 255))

    # Remove x-axis labels from the upper plot
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Set y-ticks labels
    ax1.set_yticklabels([f'{i:.2f}' for i in ax1.get_yticks()], fontsize=12, fontweight='bold')
    ax2.set_yticklabels([f'{i:.2f}' for i in ax2.get_yticks()], fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
    plt.subplots_adjust(hspace=0)  # Adjust space between the two plots
    plt.show()

    # Print the best parameters for each model
    for i, name in enumerate(model_names):
        print(f"Model: {name}, Best Parameters: {best_params_list[i]}")


# Define parameter grids for grid search
param_grids = {
    "RR": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "SVR": {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale']
    },
    "RF": {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "XGB": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
    "GB": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    "AB": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    "ET": {
        'n_estimators': [10, 20, 50],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "LGB": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_data_in_leaf': [10, 20, 50],
        'max_depth': [3, 6, 9],
    },
    "CB": {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [4, 6, 8],
    },
}

# Example usage
total_excel_file = 'data_v2_1.xlsx'  # Ensure the file path is correct
train_and_compare_models(total_excel_file, param_grids)
