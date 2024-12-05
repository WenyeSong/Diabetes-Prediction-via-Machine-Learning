
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  average_precision_score, roc_curve

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import StackingClassifier

!pip install catboost

df = pd.read_csv("/diabetes_012_health_indicators_BRFSS2015.csv")
# print(df.head())
# print(df.shape)
print()
print(df.columns[0:])

print(df['Diabetes_012'].value_counts())

predia = df.loc[df["Diabetes_012"]== 1.0]
#drop all rows with 1.0 in the target column
df = df[df.Diabetes_012 != 1.0]
#make same number of 0 and 2 in the target column
df = df.sample(frac=1).groupby('Diabetes_012').head(35346)
print(df['Diabetes_012'].value_counts())
#change label to 0 and 1
df['Diabetes_012'] = df['Diabetes_012'].replace(2.0, 1)

X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']
print(X.shape)
print(y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
print(xTrain.shape)
print(yTrain.shape)

# manually selected features based on feature importance
xTrain_mod = xTrain.loc[:,["GenHlth","BMI","Age","HighBP","HighChol","HeartDiseaseorAttack","CholCheck","Fruits","Education"]]
xTest_mod = xTest.loc[:,["GenHlth","BMI","Age","HighBP","HighChol","HeartDiseaseorAttack","CholCheck","Fruits","Education"]]

print(xTrain_mod.shape)
print(xTest_mod.shape)

#scale data
scaler = StandardScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

scaler_mod = StandardScaler()
scaler_mod.fit(xTrain_mod)
xTrain_mod = scaler_mod.transform(xTrain_mod)
xTest_mod = scaler_mod.transform(xTest_mod)

#draw heatmap
x_train_4_hm = pd.DataFrame(xTrain, columns=X.columns)
x_train_4_hm['Diabetes_binary'] = yTrain.values
x_train_4_hm = x_train_4_hm.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(x_train_4_hm, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

corr_matrix = x_train_4_hm.abs()
#select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#drop high features
to_drop = []
#convert xTrain and xTest to dataframes to drop columns
xTrain = pd.DataFrame(xTrain, columns=X.columns)
xTest = pd.DataFrame(xTest, columns=X.columns)
for col in upper.columns:
  if any(upper[col] > 0.8):
    correlated_pairs = upper.index[upper[col] > 0.8].tolist()
    drop = col if corr_matrix.loc[col, xTrain.columns[-1]] < max(corr_matrix.loc[correlated_pairs, xTrain.columns[-1]]) else correlated_pairs
    to_drop.extend(drop if isinstance(drop, list) else [drop])

xTrain.drop(columns=set(to_drop), inplace=True)
xTest.drop(columns=set(to_drop), inplace=True)

#drop low features
target = x_train_4_hm.columns[-1]
low_corr_features = corr_matrix.index[corr_matrix[target] < 0.05].tolist()
if target in low_corr_features:
  low_corr_features.remove(target)
xTrain.drop(columns=low_corr_features, inplace=True)
xTest.drop(columns=low_corr_features, inplace=True)

print("Columns to drop due to high correlation:", to_drop)
print("Columns to drop due to low correlation with target:", low_corr_features)

#plot the modified dataset
x_train_4_hm = xTrain
x_train_4_hm['Diabetes_binary'] = yTrain.values
x_train_4_hm = x_train_4_hm.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(x_train_4_hm, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

print(xTrain.head())
print(xTrain.shape)

xTrain = xTrain.drop(columns="Diabetes_binary",axis=1)
print(xTrain.columns)
print(xTest.columns)

#mutual information to target variables
xTrain = pd.DataFrame(xTrain)
xTest = pd.DataFrame(xTest)


mi_scores = mutual_info_classif(xTrain, yTrain)
mi_scores_df = pd.Series(mi_scores, index=xTrain.columns[:])

#plot mutual information scores using a bar plot
plt.figure(figsize=(12, 8))
mi_scores_df.sort_values(ascending=False).plot(kind='bar')
plt.title('Mutual Information Scores')
plt.show()

#drop low features to target based on mutual information
to_drop = mi_scores_df[mi_scores_df < 0.01].index
xTrain.drop(columns=set(to_drop), inplace=True)
xTest.drop(columns=set(to_drop), inplace=True)
print(xTrain.columns)

full_X = np.concatenate((xTrain, xTest), axis=0)
print(full_X.shape)

full_y = np.concatenate((yTrain, yTest), axis=0)

#plot the modified dataset
x_train_4_hm = xTrain
x_train_4_hm['Diabetes_binary'] = yTrain.values
x_train_4_hm = x_train_4_hm.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(x_train_4_hm, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

print(xTrain.head())
print(xTrain.shape)

xTrain = xTrain.to_numpy()
xTest = xTest.to_numpy()

def evaluate_model(model, xTest, yTest):
    y_pred = model.predict(xTest) # predict
    y_scores = model.predict_proba(xTest)[:, 1]

    #calculate metrics
    accuracy = accuracy_score(yTest, y_pred)
    precision = precision_score(yTest, y_pred, zero_division=0)
    recall = recall_score(yTest, y_pred, zero_division=0)
    f1 = f1_score(yTest, y_pred, zero_division=0)
    roc_auc = roc_auc_score(yTest, y_scores)

    #ROC curve data
    fpr, tpr, _ = roc_curve(yTest, y_scores)

    #Calibration curve data
    prob_true, prob_pred = calibration_curve(yTest, y_scores, n_bins=10)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'prob_true': prob_true,
        'prob_pred': prob_pred,
        'Confusion Matrix': confusion_matrix(yTest, y_pred)
    }

def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Random Search CV for hyperparameter tuning.
    """
    total_combinations = np.prod([len(v) for v in pgrid.values()])
    n_iter = max(10, int(total_combinations * 0.33))
    random_search = RandomizedSearchCV(clf, param_distributions=pgrid, n_iter=n_iter, scoring='roc_auc', cv=5)
    random_search.fit(xTrain, yTrain)
    best_clf = random_search.best_estimator_
    y_pred = best_clf.predict(xTest)
    y_scores = best_clf.predict_proba(xTest)[:, 1]
    best_params = random_search.best_params_

    prob_true, prob_pred = calibration_curve(yTest, y_scores, n_bins=15)
    results = {"Accuracy":accuracy_score(yTest, y_pred),
               "Precision": precision_score(yTest, y_pred, zero_division=0),
               "Recall":recall_score(yTest, y_pred, zero_division=0),
               "F1":f1_score(yTest, y_pred, zero_division=0),
               "AUC": roc_auc_score(yTest, y_scores),
               "AUPRC": average_precision_score(yTest, y_scores),
               "F1": f1_score(yTest, y_pred),
               'prob_true': prob_true,
               'prob_pred': prob_pred,
               'Confusion Matrix': confusion_matrix(yTest, y_pred)}
    fpr, tpr, _ = roc_curve(yTest, y_scores)
    roc = {"fpr": fpr, "tpr": tpr,"yPredProb": y_scores}
    return results, roc, best_params, best_clf

def get_parameter_grid(mName):
    if mName == "DT":
        return {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 10, 20, 40],
            'min_samples_leaf': [1, 2, 4, 6],
            'criterion': ['gini', 'entropy']
        }
    elif mName == "LR (L1)":
        return {
            'solver': ['liblinear', 'saga'],
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1']
        }
    elif mName == "LR (L2)":
        return {
            'solver': ['newton-cg', 'lbfgs', 'saga'],
            'C': [0.1, 1],
            'penalty': ['l2']
        }
    elif mName == "KNN":
        return {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'p': [1, 2]
        }
    elif mName == "NN":
        return {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }
    elif mName == "NB":
        return {
            'var_smoothing': [1e-8, 1e-7, 1e-6, 1e-5]
        }
    elif mName == "RF":
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    elif mName == "XGB":
        return {
            'max_depth': [6, 10, 15, 20],
            'min_child_weight': [5, 10],
            'gamma': [0, 0.1, 0.5, 1, 1.5],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.8, 0.9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2]
        }
    elif  mName == "CatBoost":
        return {
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 500, 1000]
        }

models = {
    "XGB":xgb.XGBClassifier(),
    "CatBoost":CatBoostClassifier(verbose=0),
    "DT": DecisionTreeClassifier(),
    "LR (L1)": LogisticRegression(),
    "LR (L2)": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NN": MLPClassifier(max_iter=900),
    "NB": GaussianNB(),
    "RF": RandomForestClassifier()
}


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # ROC plot
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')

plt.subplot(1, 2, 2)  # Calibration plot
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.title('Calibration Curve')


base_model = []

for mName, model in models.items():
    pGrid = get_parameter_grid(mName)
    results, roc_dict, best_params, best_clf = eval_randomsearch(model, pGrid, xTrain, yTrain, xTest, yTest)
    base_model.append(best_clf)

    print(f"{mName} Results: {results}")
    plt.subplot(1, 2, 1)
    plt.plot(roc_dict["fpr"], roc_dict["tpr"], label=f"{mName} (AUC = {results['AUC']:.2f})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.subplot(1, 2, 2)
    prob_true, prob_pred = results["prob_true"],results["prob_pred"]
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f"{mName}")
    best_para = {}
    best_para[mName] = best_params

plt.subplot(1, 2, 1)
plt.legend()
plt.subplot(1, 2, 2)
plt.xlabel('Mean predicted probability')
plt.ylabel('True probability in each bin')
plt.legend()
plt.tight_layout()
plt.show()

print(best_para)

print(base_model)

estimators = [
    ('XGB', base_model[0]),
    ('CatBoost', base_model[1]),
    ('DT', base_model[2]),
    ('LR(L1)',base_model[3]),
    ('LR(L2)',base_model[4]),
    ('KNeighborsClassifier',base_model[5]),
    ('MLPClassifier',base_model[6]),
    ('GaussianNB',base_model[7]),
    ('RandomForestClassifier',base_model[8])
]

meta_model = LogisticRegression()

stacking_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5)
stacking_model.fit(xTrain, yTrain)
stacking_score = stacking_model.score(xTest, yTest)
y_pred = stacking_model.predict(xTest)
y_proba = stacking_model.predict_proba(xTest)[:, 1]

accuracy = accuracy_score(yTest, y_pred)
f1 = f1_score(yTest, y_pred)
roc_auc = roc_auc_score(yTest, y_proba)
auprc = average_precision_score(yTest, y_proba)
precision = precision_score(yTest, y_pred)
recall = recall_score(yTest, y_pred)

print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"AUPRC: {auprc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

models = {
    "XGB":xgb.XGBClassifier(),
    "CatBoost":CatBoostClassifier(verbose=0),
    "DT": DecisionTreeClassifier(),
    "LR (L1)": LogisticRegression(),
    "LR (L2)": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NN": MLPClassifier(max_iter=900),
    "NB": GaussianNB(),
    "RF": RandomForestClassifier()
}


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # ROC plot
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')

plt.subplot(1, 2, 2)  # Calibration plot
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.title('Calibration Curve')


base_model1 = []

for mName, model in models.items():
    pGrid = get_parameter_grid(mName)
    results, roc_dict, best_params, best_clf1 = eval_randomsearch(model, pGrid, xTrain_mod, yTrain, xTest_mod, yTest)
    base_model1.append(best_clf1)

    print(f"{mName} Results: {results}")
    plt.subplot(1, 2, 1)
    plt.plot(roc_dict["fpr"], roc_dict["tpr"], label=f"{mName} (AUC = {results['AUC']:.2f})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.subplot(1, 2, 2)
    prob_true, prob_pred = results["prob_true"],results["prob_pred"]
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f"{mName}")
    best_para = {}
    best_para[mName] = best_params

plt.subplot(1, 2, 1)
plt.legend()
plt.subplot(1, 2, 2)
plt.xlabel('Mean predicted probability')
plt.ylabel('True probability in each bin')
plt.legend()
plt.tight_layout()
plt.show()

print(base_model1)

# # calculating MSE for the calibration curve

# import numpy as np

# # Provided list of arrays
# calibration = [
#     np.array([0.07830189, 0.17123857, 0.29387755, 0.39486673, 0.44464446, 0.52709725, 0.62376888, 0.72820513, 0.80536913, 0.88849765]),
#     np.array([0.04044449, 0.1403733, 0.25308884, 0.35164687, 0.45731934, 0.54844162, 0.65263838, 0.7527935, 0.8437675, 0.92977726]),
#     np.array([0.02738712, 0.12196679, 0.22264438, 0.38461538, 0.48865356, 0.57320644, 0.68417597, 0.74674556, 0.81349424, 0.89079229]),
#     np.array([0.06160259, 0.14735613, 0.24929831, 0.34849705, 0.45175613, 0.55167094, 0.6512203, 0.75028087, 0.84934358, 0.9337826]),
#     np.array([0.02718589, 0.12324393, 0.22222222, 0.38517325, 0.4890566, 0.57509158, 0.68268598, 0.7464455, 0.81404279, 0.89114194]),
#     np.array([0.06155898, 0.14750944, 0.24946595, 0.34836959, 0.45157793, 0.55171999, 0.65127106, 0.7503035, 0.84940051, 0.93385993]),
#     np.array([0.10698603, 0.20304017, 0.27733935, 0.38666667, 0.48157454, 0.56082148, 0.62643291, 0.72952854, 0.7677665, 0.79709302]),
#     np.array([0.02538073, 0.15853674, 0.25490756, 0.35423569, 0.45257893, 0.54925755, 0.64760073, 0.74396171, 0.83717172, 0.95272863]),
#     np.array([0.046716, 0.16707416, 0.26364477, 0.37177122, 0.49252492, 0.57851852, 0.66807739, 0.75204082, 0.86085701, 0.92352941]),
#     np.array([0.0482025, 0.14799431, 0.24957784, 0.34931252, 0.45182552, 0.55211795, 0.65174509, 0.75220592, 0.8479804, 0.91546184]),
#     np.array([0.18369855, 0.41121495, 0.5177305, 0.52444444, 0.60332542, 0.60382514, 0.63874346, 0.66743119, 0.64027539, 0.74807692]),
#     np.array([0.01961588, 0.14595748, 0.24767973, 0.34806777, 0.44948973, 0.54989714, 0.6516713, 0.75186332, 0.8530254, 0.98852294]),
#     np.array([0.03138686, 0.10688666, 0.22810219, 0.32887029, 0.4406657, 0.56277884, 0.66036687, 0.76358105, 0.8697001, 0.95959596]),
#     np.array([0.06004138, 0.14850408, 0.24799907, 0.35203288, 0.45123292, 0.55171822, 0.65299293, 0.75116822, 0.84579934, 0.90872831])
# ]

# for i in range(0, len(calibration), 2):
#     prob_true = calibration[i]
#     if i+1 < len(calibration):  # Ensure there is a pair
#         prob_pred = calibration[i+1]
#         mse = np.mean((prob_true - prob_pred) ** 2)
#         print(f"MSE between set {i} and {i+1}: {mse}")
#
# #DT LR(L1) LR(L2) KNN NN NB RF

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/wen/Desktop/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

#split train set and test set
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']  #target
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

#scale
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

x_train_df = pd.DataFrame(xTrain, columns=X.columns)
x_train_df['Diabetes_binary'] = yTrain.values
correlation_matrix = x_train_df.corr().abs()

#upper triangle of correlation matrix
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

#drop based on high correlation
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
xTrain = pd.DataFrame(xTrain, columns=X.columns)
xTest = pd.DataFrame(xTest, columns=X.columns)
xTrain.drop(columns=to_drop, inplace=True)
xTest.drop(columns=to_drop, inplace=True)

#drop features with low correlation to target
low_corr_features = correlation_matrix.index[correlation_matrix['Diabetes_binary'] < 0.05].tolist()
xTrain.drop(columns=low_corr_features, inplace=True)
xTest.drop(columns=low_corr_features, inplace=True)

feature_names = xTrain.columns.tolist()
print(feature_names, len(feature_names))

#convert df back to np arrays
xTrain = xTrain.values
xTest = xTest.values



# Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(xTrain, yTrain)

#predict and evaluate
yPred = rf.predict(xTest)
accuracy = accuracy_score(yTest, yPred)
print(f"Accuracy: {accuracy:.2f}")

#assess feature importance
importances = rf.feature_importances_
feature_importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
sorted_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

# print("Feature Importances:")
# for name, importance in sorted_importance:
#     print(f"{name}: {importance:.4f}")

#assess permutation importance
perm_importance = permutation_importance(rf, xTest, yTest, n_repeats=30, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()


# permutation importance box plot
plt.figure(figsize=(10, 8))
plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False,
            labels=np.array(feature_names)[sorted_idx],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor="skyblue", color="darkblue", linewidth=1.5),
            whiskerprops=dict(color="darkblue", linewidth=1.5),
            capprops=dict(color="darkblue", linewidth=1.5))
plt.title("Permutation Importances (test set)", fontsize=18, weight='bold')
plt.xlabel("Decrease in accuracy score", fontsize=14, weight='bold')
plt.ylabel("Features", fontsize=14, weight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# feature importance plot
plt.figure(figsize=(10, 8))
sns.barplot(x=[importance for _, importance in sorted_importance], y=[name for name, _ in sorted_importance])
plt.title('Feature Importance', fontsize=18, weight='bold')
plt.xlabel('Importance', fontsize=14, weight='bold')
plt.ylabel('Features', fontsize=14, weight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X = np.array(X)
y = np.array(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42, perplexity=150, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5,s=1)
plt.colorbar(scatter)
plt.title('t-SNE visualization of the dataset')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

tsne = TSNE(n_components=2, random_state=42, perplexity=100, n_iter=1000)
X_tsne = tsne.fit_transform(full_X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=full_y, cmap='viridis', alpha=0.5,s=1)
plt.colorbar(scatter)
plt.title('t-SNE visualization of the dataset')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

tsne = TSNE(n_components=2, random_state=42, perplexity=150, n_iter=1000)
X_tsne = tsne.fit_transform(full_X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=full_y, cmap='viridis', alpha=0.5,s=1)
plt.colorbar(scatter)
plt.title('t-SNE visualization of the dataset')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

tsne = TSNE(n_components=2, random_state=42, perplexity=265, n_iter=300)
X_tsne = tsne.fit_transform(full_X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=full_y, cmap='viridis', alpha=0.5,s=1)
plt.colorbar(scatter)
plt.title('t-SNE visualization of the dataset')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()





