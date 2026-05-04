"""
Complete reproducible code for the NBA supervised learning assignment.
Run this file from the project folder:
    python generate_results.py
It reads nba_dailyleaders_full_24_25.csv and saves all result tables and figures.
"""
import os, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.base import clone

BASE = Path(__file__).resolve().parent
TABLES = BASE / 'tables'
FIGS = BASE / 'figures'
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

def clean_data(df):
    data = df.copy()
    if 'Unnamed: 3' in data.columns:
        data = data.rename(columns={'Unnamed: 3': 'HomeAway'})
    data['HomeAway'] = data['HomeAway'].fillna('Home').replace('@', 'Away')
    data['Result_binary'] = data['Result'].map({'W': 1, 'L': 0})
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    def convert_minutes(x):
        try:
            m, s = str(x).split(':')
            return int(m) + int(s) / 60
        except Exception:
            return np.nan
    data['MP_decimal'] = data['MP'].apply(convert_minutes)
    return data

def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    return preprocessor, numeric_features, categorical_features

def evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor):
    pipe = Pipeline([('preprocessor', clone(preprocessor)), ('model', clone(model))])
    start = time.time()
    pipe.fit(X_train, y_train)
    training_time = time.time() - start
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return pipe, {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_prob),
        'Training Time Seconds': training_time
    }

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGS / f'{name}.pdf')
    plt.savefig(FIGS / f'{name}.png', dpi=200)
    plt.close()

# Load and prepare data
DATA_PATH = BASE / 'nba_dailyleaders_full_24_25.csv'
df = pd.read_csv(DATA_PATH)
df_clean = clean_data(df)
features = ['Tm','HomeAway','Opp','Month','DayOfWeek','MP_decimal','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','GmSc']
X = df_clean[features]
y = df_clean['Result_binary']
preprocessor, numeric_features, categorical_features = make_preprocessor(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

# Dataset summary
pd.DataFrame({'Item':['Dataset name','Rows','Columns','Target variable','Task type','Positive class','Negative class'], 'Value':['NBA Daily Leaders / NBA Player Stats 2024-25', df.shape[0], df.shape[1], 'Result', 'Binary classification', 'W = Win', 'L = Loss']}).to_csv(TABLES / 'dataset_summary.csv', index=False)
df.isna().sum().reset_index().rename(columns={'index':'Column',0:'Missing Values'}).to_csv(TABLES / 'missing_values_summary.csv', index=False)
df['Result'].value_counts().reset_index().rename(columns={'Result':'Class','count':'Count'}).to_csv(TABLES / 'class_distribution.csv', index=False)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=9),
    'Random Forest': RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, learning_rate=.07, max_depth=3, random_state=42)
}

# RQ1 and RQ2
baseline_names = ['Logistic Regression', 'Decision Tree', 'k-NN']
baseline_results, pipes = [], {}
for name in baseline_names:
    pipe, metrics = evaluate_model(name, models[name], X_train, X_test, y_train, y_test, preprocessor)
    baseline_results.append(metrics)
    pipes[name] = pipe
rq1 = pd.DataFrame(baseline_results).round(4)
rq1.to_csv(TABLES / 'rq1_baseline_model_performance.csv', index=False)
rq1.plot(x='Model', y=['Accuracy','Precision','Recall','F1-score'], kind='bar', figsize=(10,6), title='RQ1: Baseline Model Performance')
plt.ylabel('Score'); plt.xlabel(''); plt.xticks(rotation=30, ha='right'); savefig('rq1_baseline_model_performance')

all_results = baseline_results.copy()
for name in ['Random Forest', 'Gradient Boosting']:
    pipe, metrics = evaluate_model(name, models[name], X_train, X_test, y_train, y_test, preprocessor)
    all_results.append(metrics)
    pipes[name] = pipe
rq2 = pd.DataFrame(all_results).sort_values('F1-score', ascending=False).round(4)
rq2.to_csv(TABLES / 'rq2_candidate_model_comparison.csv', index=False)
plt.figure(figsize=(10,6)); plt.barh(rq2['Model'], rq2['F1-score']); plt.xlabel('F1-score'); plt.title('RQ2: Candidate Model Ranking by F1-score'); plt.gca().invert_yaxis(); savefig('rq2_candidate_model_ranking')
best_model_name = rq2.iloc[0]['Model']; best_pipe = pipes[best_model_name]

# RQ3
numeric_only = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_trans = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
pre_num_min = ColumnTransformer([('num', SimpleImputer(strategy='median'), numeric_only)])
pre_num_scaled = ColumnTransformer([('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_only)])
pre_full_no_scale = ColumnTransformer([('num', SimpleImputer(strategy='median'), numeric_features), ('cat', cat_trans, categorical_features)])
pre_full = preprocessor
rq3_rows = []
for strategy, pre, numeric_only_flag in [
    ('Numeric only + imputation', pre_num_min, True),
    ('Numeric only + imputation + scaling', pre_num_scaled, True),
    ('Numeric + categorical encoding', pre_full_no_scale, False),
    ('Full pipeline', pre_full, False)]:
    X_use = X[numeric_only] if numeric_only_flag else X
    X_tr, X_te, y_tr, y_te = train_test_split(X_use, y, test_size=.2, random_state=42, stratify=y)
    _, metrics = evaluate_model(strategy, LogisticRegression(max_iter=1000, random_state=42), X_tr, X_te, y_tr, y_te, pre)
    rq3_rows.append({'Preprocessing Strategy': strategy, **{k:v for k,v in metrics.items() if k in ['Accuracy','Precision','Recall','F1-score','AUC']}})
rq3 = pd.DataFrame(rq3_rows).round(4)
rq3.to_csv(TABLES / 'rq3_preprocessing_impact.csv', index=False)
plt.figure(figsize=(10,6)); plt.bar(rq3['Preprocessing Strategy'], rq3['F1-score']); plt.ylabel('F1-score'); plt.title('RQ3: Impact of Preprocessing on Logistic Regression'); plt.xticks(rotation=30, ha='right'); savefig('rq3_preprocessing_impact')

# RQ4
feature_pipe = Pipeline([('preprocessor', clone(preprocessor)), ('model', GradientBoostingClassifier(n_estimators=80, learning_rate=.07, max_depth=3, random_state=42))])
feature_pipe.fit(X_train, y_train)
feature_names = [n.replace('num__','').replace('cat__','') for n in feature_pipe.named_steps['preprocessor'].get_feature_names_out()]
rq4 = pd.DataFrame({'Feature': feature_names, 'Importance Score': feature_pipe.named_steps['model'].feature_importances_}).sort_values('Importance Score', ascending=False).head(15).reset_index(drop=True)
rq4.insert(0, 'Rank', range(1, len(rq4)+1)); rq4 = rq4.round(4)
rq4.to_csv(TABLES / 'rq4_feature_importance.csv', index=False)
plot = rq4.head(10).sort_values('Importance Score')
plt.figure(figsize=(10,6)); plt.barh(plot['Feature'], plot['Importance Score']); plt.xlabel('Importance Score'); plt.title('RQ4: Top 10 Feature Importances'); savefig('rq4_feature_importance')

# RQ5
metrics_list = ['Accuracy','Precision','Recall','F1-score','AUC']
rq5 = rq2[['Model'] + metrics_list].copy()
for metric in metrics_list:
    rq5[f'Rank by {metric}'] = rq5[metric].rank(ascending=False, method='min').astype(int)
rq5_rank = rq5[['Model'] + [f'Rank by {m}' for m in metrics_list]]
rq5_rank.to_csv(TABLES / 'rq5_metric_sensitivity_ranking.csv', index=False)
rank_plot = rq5_rank.set_index('Model')
plt.figure(figsize=(10,6))
for model in rank_plot.index:
    plt.plot(metrics_list, rank_plot.loc[model], marker='o', label=model)
plt.gca().invert_yaxis(); plt.ylabel('Rank (1 = Best)'); plt.title('RQ5: Model Ranking Across Evaluation Metrics'); plt.legend(bbox_to_anchor=(1.05,1), loc='upper left'); savefig('rq5_metric_sensitivity_ranking')

# RQ6
selected_model = models[best_model_name]
rq6_rows = []
def split_metrics(test_size, label):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    _, met = evaluate_model(label, selected_model, X_tr, X_te, y_tr, y_te, preprocessor)
    rq6_rows.append({'Scenario': label, **{k:v for k,v in met.items() if k in ['Accuracy','Precision','Recall','F1-score','AUC']}, 'Std. Dev.': np.nan})
split_metrics(.3, '70/30 train-test split')
split_metrics(.2, '80/20 train-test split')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_pipe = Pipeline([('preprocessor', clone(preprocessor)), ('model', clone(selected_model))])
cv_f1 = cross_val_score(cv_pipe, X, y, cv=cv, scoring='f1', n_jobs=1)
cv_acc = cross_val_score(cv_pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)
rq6_rows.append({'Scenario':'5-fold cross-validation','Accuracy':cv_acc.mean(),'Precision':np.nan,'Recall':np.nan,'F1-score':cv_f1.mean(),'AUC':np.nan,'Std. Dev.':cv_f1.std()})
rng = np.random.default_rng(42)
def add_noise(d, cols, level=.1):
    d = d.copy()
    for col in cols:
        std = d[col].std(skipna=True)
        if pd.notna(std) and std > 0:
            d[col] = d[col] + rng.normal(0, level*std, size=len(d))
    return d
def add_missing(d, cols, rate=.2):
    d = d.copy(); mask = rng.random((len(d), len(cols))) < rate
    for i, col in enumerate(cols):
        d.loc[mask[:, i], col] = np.nan
    return d
_, noise = evaluate_model('10% numeric noise added', selected_model, add_noise(X_train, numeric_features), add_noise(X_test, numeric_features), y_train, y_test, preprocessor)
rq6_rows.append({'Scenario':'10% numeric noise added', **{k:v for k,v in noise.items() if k in ['Accuracy','Precision','Recall','F1-score','AUC']}, 'Std. Dev.':np.nan})
_, missing = evaluate_model('20% numeric missingness', selected_model, add_missing(X_train, numeric_features), add_missing(X_test, numeric_features), y_train, y_test, preprocessor)
rq6_rows.append({'Scenario':'20% numeric missingness', **{k:v for k,v in missing.items() if k in ['Accuracy','Precision','Recall','F1-score','AUC']}, 'Std. Dev.':np.nan})
rq6 = pd.DataFrame(rq6_rows).round(4)
rq6.to_csv(TABLES / 'rq6_robustness_analysis.csv', index=False)
plt.figure(figsize=(10,6)); plt.bar(rq6['Scenario'], rq6['F1-score']); plt.ylabel('F1-score'); plt.title(f'RQ6: Robustness Analysis of {best_model_name}'); plt.xticks(rotation=30, ha='right'); savefig('rq6_robustness_analysis')

# RQ7 and confusion matrix
interpretability = {'Logistic Regression':'High','Decision Tree':'High','k-NN':'Low','Random Forest':'Medium','Gradient Boosting':'Medium-Low'}
def performance_label(f1):
    return 'Very High' if f1 >= .75 else 'High' if f1 >= .70 else 'Medium' if f1 >= .60 else 'Low'
def cost_label(seconds):
    return 'Low' if seconds < .5 else 'Medium' if seconds < 3 else 'High'
rq7_rows = []
for _, row in rq2.iterrows():
    model = row['Model']
    rq7_rows.append({'Model':model,'Predictive Performance':performance_label(row['F1-score']),'Interpretability':interpretability.get(model,'Medium'),'AUC':row['AUC'],'F1-score':row['F1-score'],'Computational Cost':cost_label(row['Training Time Seconds']),'Practical Usefulness':'High' if model in ['Logistic Regression','Random Forest','Gradient Boosting'] else 'Medium','Final Recommendation':'Selected' if model == best_model_name else 'Not selected'})
rq7 = pd.DataFrame(rq7_rows)
rq7.to_csv(TABLES / 'rq7_final_decision_matrix.csv', index=False)
plot = rq7.set_index('Model')[['F1-score','AUC']]
plt.figure(figsize=(10,6)); width=.35; xs=np.arange(len(plot.index)); plt.bar(xs-width/2, plot['F1-score'], width, label='F1-score'); plt.bar(xs+width/2, plot['AUC'], width, label='AUC'); plt.xticks(xs, plot.index, rotation=30, ha='right'); plt.ylabel('Score'); plt.title('RQ7: Final Model Trade-off Comparison'); plt.legend(); savefig('rq7_final_model_tradeoff')
final_pred = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, final_pred)
pd.DataFrame(cm, index=['Actual Loss','Actual Win'], columns=['Predicted Loss','Predicted Win']).to_csv(TABLES / 'final_model_confusion_matrix.csv')
fig, ax = plt.subplots(figsize=(6,5)); ax.imshow(cm); ax.set_title(f'Confusion Matrix: {best_model_name}'); ax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); ax.set_xticks([0,1], ['Loss','Win']); ax.set_yticks([0,1], ['Loss','Win'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center')
fig.tight_layout(); fig.savefig(FIGS / 'final_model_confusion_matrix.pdf'); fig.savefig(FIGS / 'final_model_confusion_matrix.png', dpi=200); plt.close()
pd.DataFrame({'Item':['Best model based on F1-score','Best F1-score','Best AUC','Main note'], 'Value':[best_model_name, rq2.iloc[0]['F1-score'], rq2.iloc[0]['AUC'], 'The main model excludes +/- to reduce outcome leakage.']}).to_csv(TABLES / 'final_summary.csv', index=False)
print('All tables and figures generated successfully.')
