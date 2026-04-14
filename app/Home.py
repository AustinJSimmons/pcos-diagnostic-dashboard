import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from styles import apply_styles

st.set_page_config(
    page_title="PCOS Diagnostic Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_styles()

# Title — emoji separated from gradient span so it renders correctly
st.markdown(
    '<p style="font-size:2.4rem; font-weight:700; margin-bottom:4px;">'
    '<span style="color:#EA288D;">PCOS Diagnostic Dashboard</span></p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color:#888; font-size:1.1rem; margin-top:-8px; margin-bottom:24px;">'
    'Explore PCOS phenotypes, calculate risk, and visualize feature impacts</p>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### About This Dashboard

    Polycystic Ovary Syndrome (PCOS) is one of the most common endocrine disorders in reproductive-age
    women, affecting an estimated 8–13% of this population globally — yet up to 70% of cases remain
    undiagnosed. PCOS presents heterogeneously: some patients present primarily with metabolic features
    (insulin resistance, obesity, dyslipidemia), while others show a predominantly hormonal or
    reproductive profile (elevated androgens, anovulation, polycystic ovarian morphology). This
    variability makes early, consistent screening difficult.

    This dashboard was built to make data-driven PCOS assessment more accessible. Using a clinical
    dataset of 541 patients (177 PCOS, 364 controls) spanning 41 features — including hormonal assays,
    anthropometric measurements, ultrasound findings, and self-reported symptoms — it provides three
    complementary tools:

    - **Phenotype Explorer**: Unsupervised clustering (K-Means + PCA) to identify distinct PCOS subtypes
    - **Risk Calculator**: Logistic Regression and XGBoost models for personalized risk scoring, with both a full clinical model and a non-invasive screening model
    - **Feature Impact**: Feature importance analysis and head-to-head model comparison across algorithms and feature sets

    The non-invasive model is particularly relevant for low-resource or primary care settings, where
    laboratory tests and ultrasound may not be immediately available. It relies solely on symptoms,
    vitals, and lifestyle factors — enabling a first-pass screening without any invasive workup.

    **Dataset:** Sourced from a fertility clinic dataset, pre-processed and cleaned for analysis.
    **Methods:** K-Means clustering, logistic regression, XGBoost with RandomizedSearchCV tuning, SHAP analysis.
    """)

with col2:
    st.markdown("### Quick Statistics")

    possible_paths = [
        Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_data.csv',
        Path('data') / 'processed' / 'cleaned_data.csv'
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    try:
        if data_path is None:
            raise FileNotFoundError("Could not find cleaned_data.csv")
        df = pd.read_csv(data_path)

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Total Patients", f"{len(df):,}")
            st.metric("PCOS Cases", f"{(df['pcos_y_n'] == 1).sum()}")
            st.metric("Clinical Features", "41")
        with metric_col2:
            pcos_pct = (df['pcos_y_n'] == 1).sum() / len(df) * 100
            st.metric("PCOS Prevalence", f"{pcos_pct:.1f}%")
            st.metric("Control Group", f"{(df['pcos_y_n'] == 0).sum()}")
            st.metric("Best Model AUC", "94.7%")
    except:
        st.warning("Dataset not found. Please check data path.")

    st.divider()
    st.markdown("""
    ### Limitations & Disclaimer

    This tool is intended for **research and educational purposes only**. Risk scores produced
    by these models should not be used as a substitute for clinical judgement, formal diagnostic
    criteria (Rotterdam criteria), or evaluation by a qualified healthcare provider.

    Model performance was validated on a held-out test set from a single clinical dataset and
    may not generalize to all populations.
    """)

st.divider()

# --- Research Context ---
st.markdown("### Research Context")

ctx_col1, ctx_col2 = st.columns(2)

with ctx_col1:
    st.markdown("""
    **Project:** Predicting and Subtyping Polycystic Ovary Syndrome: A Machine Learning Approach to Non-Invasive Screening and Phenotypic Clustering

    **Course:** DATA 501 – University of Calgary, Winter 2026

    **Team:** Austin Simmons · Naima Noor · Sarthak Monga
    """)

with ctx_col2:
    st.markdown("""
    **RQ1:** Which low-cost, non-invasive clinical indicators are the strongest predictors of PCOS in the absence of ultrasound and blood test data?

    **RQ2:** What is the smallest set of variables that can reliably screen for PCOS while maintaining acceptable diagnostic performance?

    **RQ3:** To what extent can cycle irregularity and cycle length predict a PCOS diagnosis?
    """)

st.divider()

# --- Motivation & Background ---
st.markdown("### Motivation & Background")

mot_col1, mot_col2 = st.columns(2)

with mot_col1:
    st.markdown("""
    Polycystic Ovary Syndrome is the most common endocrine disorder affecting people of reproductive
    age, yet the average time from symptom onset to diagnosis is **2–3 years** — with many patients
    receiving incorrect or delayed diagnoses. The consequences extend well beyond reproductive health:
    PCOS is associated with elevated risk of type 2 diabetes, cardiovascular disease, endometrial
    cancer, and mental health conditions including anxiety and depression.

    The clinical gold standard — the **Rotterdam Criteria** — requires at least two of three
    conditions: oligo-ovulation, hyperandrogenism, and polycystic ovarian morphology on ultrasound.
    Two of these three criteria require specialized equipment (ultrasound imaging and hormonal blood
    panels) that are costly and unavailable in many lower-resource healthcare settings. This structural
    barrier is a primary driver of underdiagnosis and motivates the search for accessible,
    symptom-based screening approaches.
    """)

with mot_col2:
    st.markdown("""
    This project is directly informed by two recent publications:

    **Agirsoy & Oehlschlaeger (2025)** demonstrated that meaningful predictive performance for PCOS
    diagnosis can be achieved using feature subsets aligned with the Rotterdam diagnostic criteria,
    supporting a feature-subset comparison approach and the use of XGBoost with SHAP interpretability.

    **Gao et al. (2025)** applied data-driven subtyping methods to a large PCOS clinical cohort,
    revealing biologically and clinically distinct subtypes across metabolic and hormonal dimensions.
    This challenges the conventional treatment of PCOS as a homogeneous condition and motivated the
    inclusion of unsupervised clustering alongside binary classification in this project.

    Together, these lines of evidence support the hypothesis that machine learning applied to
    low-cost clinical data can serve as a viable first-line screening tool, and that phenotypic
    subtyping provides additional clinical value beyond a binary prediction.
    """)

st.divider()

# --- Key Findings ---
st.markdown("### Key Findings at a Glance")

kf_col1, kf_col2, kf_col3 = st.columns(3)

with kf_col1:
    st.markdown("#### Model Performance")
    st.metric("Full Clinical AUC-ROC (XGBoost)", "94.8%")
    st.metric("Non-Invasive AUC-ROC (XGBoost)", "88.1%")
    st.metric("Diagnostic Cost of Removing Labs", "~7 pp gap")
    st.markdown("""
    The ~7 percentage-point gap between full clinical and non-invasive conditions quantifies
    the informational cost of removing ultrasound and blood test data — a relatively modest gap,
    suggesting symptom-based screening carries substantial predictive signal.
    """)

with kf_col2:
    st.markdown("#### Minimal Feature Set (SHAP)")
    st.metric("Stable Features Identified", "10")
    st.metric("Non-Invasive Among Them", "7 of 10")
    st.markdown("""
    SHAP stability analysis across 5 cross-validation folds identified a preliminary minimal
    feature set. The most important non-invasive predictors were:
    - Hair growth (hirsutism)
    - Skin darkening
    - Weight gain
    - Cycle irregularity & cycle length
    - Pimples / acne
    """)

with kf_col3:
    st.markdown("#### Phenotypic Clustering")
    st.metric("Phenotypes Identified", "2")
    st.metric("Metabolic Subtype", "n = 59  |  BMI 29.1")
    st.metric("Hyperandrogenic Subtype", "n = 118  |  BMI 23.7")
    st.markdown("""
    K-Means clustering (k=2) on the PCOS-positive cohort revealed two biologically distinct
    subtypes consistent with Gao et al. (2025). Both phenotypes share similar follicle counts
    and cycle irregularity rates — the distinguishing signal is **metabolic vs. hormonal**.
    """)

st.divider()

st.markdown("### Navigation Guide")

guide_col1, guide_col2, guide_col3 = st.columns(3)

with guide_col1:
    st.markdown("""
    #### Phenotype Explorer
    Analyze clustering results and explore distinct PCOS phenotypes:
    - View 2D PCA projection of clusters
    - Compare phenotype characteristics
    - Understand metabolic vs hormonal presentations
    """)
    st.page_link("pages/1_Phenotype_Explorer.py", label="Open Phenotype Explorer →")

with guide_col2:
    st.markdown("""
    #### Risk Calculator
    Calculate PCOS risk based on patient input:
    - Choose between Logistic Regression and XGBoost
    - Full clinical model or non-invasive screening model
    - Get personalized risk score and recommendations
    """)
    st.page_link("pages/2_Risk_Calculator.py", label="Open Risk Calculator →")

with guide_col3:
    st.markdown("""
    #### Feature Impact
    Visualize feature importance and compare models:
    - Feature rankings by model coefficient and mutual information
    - Logistic Regression vs XGBoost comparison
    - Full vs non-invasive feature set performance
    """)
    st.page_link("pages/3_Feature_Impact.py", label="Open Feature Impact →")

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.markdown("**Data Source**\n\nPCOS patient dataset (541 patients, 41 features) from a fertility clinic")

with footer_col2:
    st.markdown("**Methods**\n\nK-Means clustering, Logistic Regression, XGBoost (RandomizedSearchCV), SHAP analysis")
