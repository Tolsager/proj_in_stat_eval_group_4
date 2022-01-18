import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

seed_everything(1)
df = pd.read_csv('df3.csv')
df = df.drop(416, axis=0)
df = df.sample(frac=1)
df = df[df['experiment'] != 16]

df_train, df_test = train_test_split(df, stratify=df['experiment'], test_size=0.2)
X = df_train.loc[:, 'x1':'z100'].values
y = df_train['experiment'].values

sc = StandardScaler()


def hparam_search(X, y, cv=10):
    param_grid_gb = [
        {'n_estimators': [100, 150, 200]
         , 'learning_rate':[0.001, 0.005, 0.01]
         , 'subsample': [0.5, 0.8, 1]
         , 'max_depth': [2, 3, 4, 8]
         }
    ]

    parameter_combinations = ParameterGrid(param_grid_gb)

    kf = StratifiedKFold(cv)
    split_indices = [(train_indices, test_indices) for train_indices, test_indices in kf.split(X, y)]

    best_mean_val_acc = 0
    best_params = None
    for parameters in parameter_combinations:
        print("Parameters:")
        print(parameters)
        mean_val_acc = 0
        for split in range(cv):
            gb = GradientBoostingClassifier(**parameters)
            train_indices = split_indices[split][0]
            test_indices = split_indices[split][1]
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            gb.fit(X_train, y_train)
            train_score = gb.score(X_train, y_train)
            validation_score = gb.score(X_test, y_test)

            mean_val_acc += validation_score

            print()
            print(f"    Split {split+1}")
            print(f"    Training acc: {train_score}")
            print(f"    Validation acc: {validation_score}")
            if validation_score < 0.4:
                print()
                print("    Validation acc < 0.4: Moving on to next parameters")
                break
            if split == cv - 1:
                mean_val_acc /= cv
                print(f"    Mean validation acc: {mean_val_acc}")
                if mean_val_acc > best_mean_val_acc:
                    print("    New best model found!")
                    best_mean_val_acc = mean_val_acc
                    best_params = parameters
        print()
        print()

    print("Best mean validation accuracy")
    print(best_mean_val_acc)
    print("Best parameters")
    print(best_params)


hparam_search(X, y, cv=10)
