"""
Core logic tests for PCOS Diagnostic Dashboard.
Run with: python -m pytest tests/test_dashboard.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_data.csv'

FULL_FEATURES = ['age_yrs', 'bmi', 'follicle_no_r', 'follicle_no_l', 'amhng_ml',
                 'lh_miu_ml', 'fsh_miu_ml', 'weight_kg', 'waist_hip_ratio']

NONINVASIVE_FEATURES = ['age_yrs', 'bmi', 'weight_kg', 'waist_hip_ratio',
                        'pulse_ratebpm', 'rr_breaths_min', 'cycle_lengthdays',
                        'bp_systolic_mmhg', 'bp_diastolic_mmhg', 'weight_gain_y_n',
                        'hair_growth_y_n', 'skin_darkening_y_n', 'hair_loss_y_n',
                        'pimples_y_n', 'fast_food_y_n', 'reg_exercise_y_n', 'pregnant_y_n']

CLUSTER_FEATURES = ['age_yrs', 'weight_kg', 'heightcm', 'bmi', 'pulse_ratebpm', 'rr_breaths_min',
                    'hbg_dl', 'cycle_lengthdays', 'fsh_miu_ml', 'lh_miu_ml', 'fsh_lh',
                    'hipinch', 'waistinch', 'waist_hip_ratio', 'tsh_miu_l', 'amhng_ml',
                    'prlng_ml', 'vit_d3_ng_ml', 'prgng_ml', 'rbsmg_dl', 'bp_systolic_mmhg',
                    'bp_diastolic_mmhg', 'follicle_no_l', 'follicle_no_r',
                    'avg_f_size_l_mm', 'avg_f_size_r_mm', 'endometrium_mm']


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def df():
    assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"
    return pd.read_csv(DATA_PATH)

@pytest.fixture(scope="session")
def df_pcos(df):
    return df[df['pcos_y_n'] == 1].reset_index(drop=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def test_data_file_exists():
    assert DATA_PATH.exists()

def test_data_has_required_columns(df):
    missing = [c for c in FULL_FEATURES + ['pcos_y_n'] if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def test_data_has_noninvasive_columns(df):
    missing = [c for c in NONINVASIVE_FEATURES if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def test_data_has_cluster_columns(df):
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def test_target_column_is_binary(df):
    assert set(df['pcos_y_n'].dropna().unique()).issubset({0, 1})

def test_no_all_null_columns(df):
    all_null = [c for c in df.columns if df[c].isnull().all()]
    assert not all_null, f"Completely null columns: {all_null}"


# ── Risk Calculator — model training ─────────────────────────────────────────

def test_logreg_full_trains(df):
    X = df[FULL_FEATURES].fillna(df[FULL_FEATURES].mean())
    y = df['pcos_y_n']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    assert hasattr(model, 'coef_')

def test_logreg_noninvasive_trains(df):
    X = df[NONINVASIVE_FEATURES].fillna(df[NONINVASIVE_FEATURES].mean())
    y = df['pcos_y_n']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    assert hasattr(model, 'coef_')

def test_xgb_full_trains(df):
    X = df[FULL_FEATURES].fillna(df[FULL_FEATURES].mean())
    y = df['pcos_y_n']
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight,
                          n_estimators=50, verbosity=0, eval_metric='logloss')
    model.fit(X_scaled, y)
    assert hasattr(model, 'feature_importances_')

def test_xgb_noninvasive_trains(df):
    X = df[NONINVASIVE_FEATURES].fillna(df[NONINVASIVE_FEATURES].mean())
    y = df['pcos_y_n']
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight,
                          n_estimators=50, verbosity=0, eval_metric='logloss')
    model.fit(X_scaled, y)
    assert hasattr(model, 'feature_importances_')


# ── Risk Calculator — predictions ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def logreg_full(df):
    X = df[FULL_FEATURES].fillna(df[FULL_FEATURES].mean())
    y = df['pcos_y_n']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler

def test_logreg_prediction_in_range(logreg_full):
    model, scaler = logreg_full
    sample = np.array([[28, 24.0, 10, 10, 5.0, 5.0, 6.0, 65.0, 0.85]])
    prob = model.predict_proba(scaler.transform(sample))[0][1]
    assert 0.0 <= prob <= 1.0

def test_logreg_high_risk_patient(logreg_full):
    """Patient with many PCOS indicators should score higher than a healthy patient."""
    model, scaler = logreg_full
    high_risk = np.array([[25, 32.0, 18, 18, 10.0, 15.0, 4.0, 85.0, 1.0]])
    low_risk  = np.array([[30, 20.0,  4,  4,  1.5,  3.0, 8.0, 55.0, 0.75]])
    prob_high = model.predict_proba(scaler.transform(high_risk))[0][1]
    prob_low  = model.predict_proba(scaler.transform(low_risk))[0][1]
    assert prob_high > prob_low

def test_contributions_length_matches_features(logreg_full):
    model, scaler = logreg_full
    sample = np.array([[28, 24.0, 10, 10, 5.0, 5.0, 6.0, 65.0, 0.85]])
    input_scaled = scaler.transform(sample)[0]
    contribs = model.coef_[0] * input_scaled
    assert len(contribs) == len(FULL_FEATURES)


# ── Phenotype Explorer — clustering ──────────────────────────────────────────

@pytest.fixture(scope="session")
def clustering_result(df_pcos):
    X = df_pcos[CLUSTER_FEATURES].fillna(df_pcos[CLUSTER_FEATURES].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return clusters, kmeans, pca, scaler, X_scaled, X_pca

def test_clustering_produces_two_clusters(clustering_result):
    clusters, *_ = clustering_result
    assert set(clusters) == {0, 1}

def test_both_clusters_non_empty(clustering_result):
    clusters, *_ = clustering_result
    assert (clusters == 0).sum() > 0
    assert (clusters == 1).sum() > 0

def test_pca_explains_variance(clustering_result):
    _, _, pca, *_ = clustering_result
    assert pca.explained_variance_ratio_.sum() > 0.1

def test_phenotype_naming_logic(df_pcos, clustering_result):
    clusters, *_ = clustering_result
    bmi_0 = df_pcos[clusters == 0]['bmi'].mean()
    bmi_1 = df_pcos[clusters == 1]['bmi'].mean()
    if bmi_0 > bmi_1:
        names = {0: "Metabolic PCOS", 1: "Lean PCOS"}
    else:
        names = {0: "Lean PCOS", 1: "Metabolic PCOS"}
    assert "Metabolic PCOS" in names.values()
    assert "Lean PCOS" in names.values()

def test_classify_patient_returns_valid_cluster(clustering_result, df_pcos):
    clusters, kmeans, pca, scaler, *_ = clustering_result
    sample_means = df_pcos[CLUSTER_FEATURES].mean().values.reshape(1, -1)
    sample_scaled = scaler.transform(sample_means)
    predicted = kmeans.predict(sample_scaled)[0]
    assert predicted in {0, 1}
