"""
Modeling script for Utah Fitness Business Thriving Prediction.
Equivalent to 03_modeling.ipynb but runs as a plain Python script.

Tasks:
  A. Binary classifier: predict is_thriving (1=thriving, 0=not)
  B. Regressor: predict success_score (continuous, open businesses)
  C. K-means clustering: segment Utah zip codes into market types

Outputs: figures/03_*.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, silhouette_score
)

# ── Load data ──────────────────────────────────────────────────────────────────
clf_df = pd.read_csv('data/features_classifier.csv', dtype={'zip_code': str})
reg_df = pd.read_csv('data/features_regressor.csv', dtype={'zip_code': str})
print(f'Classifier: {len(clf_df)} rows | Thriving: {clf_df["is_thriving"].sum():.0f}')
print(f'Regressor:  {len(reg_df)} rows')

# ── Feature config ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    'median_income', 'median_home_value', 'median_age',
    'pct_bachelors', 'pct_prime_gym_age', 'total_pop',
    'competition_1km', 'competition_3km',
    'market_gap', 'gym_density_per_1k', 'income_per_competitor',
    'price',
]
CATEGORICAL_FEATURES = ['category_group']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES),
])

# ══════════════════════════════════════════════════════════════════════════════
# TASK A: BINARY CLASSIFIER — Will this business thrive?
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('TASK A: Binary Classification (is_thriving)')
print('='*60)

X_clf = clf_df[ALL_FEATURES]
y_clf = clf_df['is_thriving'].astype(int)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f'Train: {len(X_train_c)} | Test: {len(X_test_c)} | Balance: {y_train_c.mean():.2f}')

results_clf = {}

# A1. Logistic Regression
print('\n--- Logistic Regression ---')
lr_pipe = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))])
lr_grid = GridSearchCV(lr_pipe, {'model__C': [0.01, 0.1, 1.0, 10.0]}, cv=5, scoring='roc_auc', n_jobs=-1)
lr_grid.fit(X_train_c, y_train_c)
lr_best = lr_grid.best_estimator_
y_pred_lr = lr_best.predict(X_test_c)
y_prob_lr = lr_best.predict_proba(X_test_c)[:, 1]
lr_auc = roc_auc_score(y_test_c, y_prob_lr)
lr_acc = accuracy_score(y_test_c, y_pred_lr)
results_clf['Logistic Regression'] = {'auc': lr_auc, 'acc': lr_acc, 'proba': y_prob_lr}
print(f'Best C={lr_grid.best_params_["model__C"]} | AUC={lr_auc:.3f} | Acc={lr_acc:.3f}')
print(classification_report(y_test_c, y_pred_lr, target_names=['Not Thriving', 'Thriving']))

# A2. Random Forest
print('\n--- Random Forest Classifier ---')
rf_c_pipe = Pipeline([('pre', preprocessor), ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))])
rf_c_grid = GridSearchCV(rf_c_pipe, {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None],
    'model__min_samples_split': [2, 5],
}, cv=5, scoring='roc_auc', n_jobs=-1)
rf_c_grid.fit(X_train_c, y_train_c)
rf_c_best = rf_c_grid.best_estimator_
y_pred_rf_c = rf_c_best.predict(X_test_c)
y_prob_rf_c = rf_c_best.predict_proba(X_test_c)[:, 1]
rf_c_auc = roc_auc_score(y_test_c, y_prob_rf_c)
rf_c_acc = accuracy_score(y_test_c, y_pred_rf_c)
results_clf['Random Forest'] = {'auc': rf_c_auc, 'acc': rf_c_acc, 'proba': y_prob_rf_c}
print(f'Best={rf_c_grid.best_params_} | AUC={rf_c_auc:.3f} | Acc={rf_c_acc:.3f}')
print(classification_report(y_test_c, y_pred_rf_c, target_names=['Not Thriving', 'Thriving']))

# A3. Gradient Boosting
print('\n--- Gradient Boosting Classifier ---')
gb_c_pipe = Pipeline([('pre', preprocessor), ('model', GradientBoostingClassifier(random_state=42))])
gb_c_grid = GridSearchCV(gb_c_pipe, {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5],
}, cv=5, scoring='roc_auc', n_jobs=-1)
gb_c_grid.fit(X_train_c, y_train_c)
gb_c_best = gb_c_grid.best_estimator_
y_pred_gb_c = gb_c_best.predict(X_test_c)
y_prob_gb_c = gb_c_best.predict_proba(X_test_c)[:, 1]
gb_c_auc = roc_auc_score(y_test_c, y_prob_gb_c)
gb_c_acc = accuracy_score(y_test_c, y_pred_gb_c)
results_clf['Gradient Boosting'] = {'auc': gb_c_auc, 'acc': gb_c_acc, 'proba': y_prob_gb_c}
print(f'Best={gb_c_grid.best_params_} | AUC={gb_c_auc:.3f} | Acc={gb_c_acc:.3f}')
print(classification_report(y_test_c, y_pred_gb_c, target_names=['Not Thriving', 'Thriving']))

# ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results_clf.items():
    fpr, tpr, _ = roc_curve(y_test_c, res['proba'])
    ax.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})')
ax.plot([0,1],[0,1],'k--', label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Is This Business Thriving?')
ax.legend(); plt.tight_layout()
plt.savefig('figures/03a_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved figures/03a_roc_curves.png')

# Feature importance (best classifier)
best_clf_name = max(results_clf, key=lambda k: results_clf[k]['auc'])
best_clf_model = {'Logistic Regression': lr_best, 'Random Forest': rf_c_best, 'Gradient Boosting': gb_c_best}[best_clf_name]
print(f'\nBest classifier: {best_clf_name}')

ohe_features = (
    best_clf_model.named_steps['pre']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(CATEGORICAL_FEATURES)
).tolist()
all_feat_names = NUMERIC_FEATURES + ohe_features
importances = best_clf_model.named_steps['model'].feature_importances_
fi_df = pd.DataFrame({'feature': all_feat_names, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=False).head(15)

print('\nTop 15 Feature Importances (Classifier):')
print(fi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1], color='steelblue')
ax.set_title(f'Top 15 Features ({best_clf_name})')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('figures/03a_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved figures/03a_feature_importance.png')

# ══════════════════════════════════════════════════════════════════════════════
# TASK B: REGRESSOR — How successful will an open business be?
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('TASK B: Regression (success_score)')
print('='*60)

X_reg = reg_df[ALL_FEATURES]
y_reg = reg_df['success_score']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

results_reg = {}

for name, pipe, params in [
    ('Ridge', Pipeline([('pre', preprocessor), ('model', Ridge())]),
     {'model__alpha': [0.1, 1.0, 10.0, 100.0]}),
    ('Random Forest', Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
     {'model__n_estimators': [100, 200], 'model__max_depth': [5, 10, None], 'model__min_samples_split': [2, 5]}),
    ('Gradient Boosting', Pipeline([('pre', preprocessor), ('model', GradientBoostingRegressor(random_state=42))]),
     {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [3, 5]}),
]:
    print(f'\n--- {name} ---')
    grid = GridSearchCV(pipe, params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_r, y_train_r)
    best = grid.best_estimator_
    y_pred = best.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
    r2   = r2_score(y_test_r, y_pred)
    results_reg[name] = {'rmse': rmse, 'r2': r2, 'model': best}
    print(f'Best={grid.best_params_} | RMSE={rmse:.4f} | R²={r2:.4f}')

reg_summary = pd.DataFrame([
    {'Model': k, 'RMSE': v['rmse'], 'R²': v['r2']} for k, v in results_reg.items()
]).round(4)
print('\nRegression Summary:')
print(reg_summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
reg_summary.plot.bar(x='Model', y='RMSE', ax=axes[0], color='tomato', legend=False)
axes[0].set_title('RMSE (lower = better)'); axes[0].set_xticklabels(reg_summary['Model'], rotation=20, ha='right')
reg_summary.plot.bar(x='Model', y='R²', ax=axes[1], color='steelblue', legend=False)
axes[1].set_title('R² (higher = better)'); axes[1].set_xticklabels(reg_summary['Model'], rotation=20, ha='right')
plt.tight_layout()
plt.savefig('figures/03b_regression_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved figures/03b_regression_comparison.png')

# Feature importance for best regressor
best_reg_name = min(results_reg, key=lambda k: results_reg[k]['rmse'])
best_reg_model = results_reg[best_reg_name]['model']
print(f'\nBest regressor: {best_reg_name}')

ohe_features = (
    best_reg_model.named_steps['pre']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(CATEGORICAL_FEATURES)
).tolist()
all_feat_names = NUMERIC_FEATURES + ohe_features
importances = best_reg_model.named_steps['model'].feature_importances_
fi_reg = pd.DataFrame({'feature': all_feat_names, 'importance': importances})
fi_reg = fi_reg.sort_values('importance', ascending=False).head(15)
print('\nTop 15 Feature Importances (Regressor):')
print(fi_reg.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(fi_reg['feature'][::-1], fi_reg['importance'][::-1], color='tomato')
ax.set_title(f'Top 15 Features ({best_reg_name} Regressor)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('figures/03b_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved figures/03b_feature_importance.png')

# ══════════════════════════════════════════════════════════════════════════════
# TASK C: K-MEANS CLUSTERING — Market segmentation by zip code
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('TASK C: K-Means Clustering (zip-code market segments)')
print('='*60)

full_df = pd.read_csv('data/utah_fitness_v2.csv', dtype={'zip_code': str})

zip_features = full_df.groupby('zip_code').agg(
    median_income=('median_income', 'first'),
    median_home_value=('median_home_value', 'first'),
    pct_prime_gym_age=('pct_prime_gym_age', 'first'),
    pct_bachelors=('pct_bachelors', 'first'),
    total_pop=('total_pop', 'first'),
    gyms_in_zip=('gyms_in_zip', 'first'),
    market_gap=('market_gap', 'first'),
    avg_rating=('rating', lambda x: x[x > 0].mean()),
    lat=('latitude', 'mean'),
    lon=('longitude', 'mean'),
).dropna().reset_index()

CLUSTER_FEATURES = [
    'median_income', 'median_home_value', 'pct_prime_gym_age',
    'pct_bachelors', 'total_pop', 'gyms_in_zip', 'market_gap', 'avg_rating',
]

scaler = StandardScaler()
X_clust = scaler.fit_transform(zip_features[CLUSTER_FEATURES].fillna(0))

# Elbow + silhouette
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_clust, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), inertias, 'o-', color='steelblue')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia'); axes[0].set_title('Elbow Method')
axes[1].plot(list(K_range), sil_scores, 'o-', color='tomato')
axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette Score'); axes[1].set_title('Silhouette Score')
plt.tight_layout()
plt.savefig('figures/03c_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved figures/03c_elbow_silhouette.png')
print('Silhouette scores:', {k: round(s, 3) for k, s in zip(K_range, sil_scores)})

best_k = max(zip(K_range, sil_scores), key=lambda x: x[1])[0]
print(f'Best K by silhouette: {best_k}')

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
zip_features['cluster'] = km_final.fit_predict(X_clust)

cluster_profile = zip_features.groupby('cluster')[CLUSTER_FEATURES].mean().round(2)
print('\nCluster profiles:')
print(cluster_profile.T)

# Map
colors = ['steelblue', 'tomato', 'seagreen', 'orange', 'purple', 'brown']
fig, ax = plt.subplots(figsize=(8, 10))
for cid in sorted(zip_features['cluster'].unique()):
    sub = zip_features[zip_features['cluster'] == cid]
    ax.scatter(sub['lon'], sub['lat'], c=colors[cid], label=f'Cluster {cid}',
               s=sub['total_pop'] / 500, alpha=0.7, edgecolors='white', linewidths=0.5)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title(f'Utah Fitness Market Segments (K={best_k})\npoint size = population')
ax.legend(); plt.tight_layout()
plt.savefig('figures/03c_utah_clusters_map.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved figures/03c_utah_clusters_map.png')

# Thriving rate per cluster
full_df = full_df.merge(zip_features[['zip_code', 'cluster']], on='zip_code', how='left')
thriving_by_cluster = (
    full_df[full_df['is_thriving'].notna()]
    .groupby('cluster')['is_thriving']
    .agg(['mean', 'count'])
    .rename(columns={'mean': 'pct_thriving', 'count': 'n'})
    .round(3)
)
print('\nThriving rate by cluster:')
print(thriving_by_cluster)

# ── Final summary ──────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('FINAL SUMMARY')
print('='*60)
print('\nClassifier results:')
for name, res in results_clf.items():
    print(f'  {name:25s} AUC={res["auc"]:.3f}  Acc={res["acc"]:.3f}')
print('\nRegressor results:')
for name, res in results_reg.items():
    print(f'  {name:25s} RMSE={res["rmse"]:.4f}  R²={res["r2"]:.4f}')
print(f'\nClustering: K={best_k}, Silhouette={max(sil_scores):.3f}')
print('\nAll figures saved to figures/')
