import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
import sys
sys.path.append(str(Path(__file__).parent.parent))
from styles import apply_styles, style_fig

st.set_page_config(
    page_title="Feature Impact Analysis",
    layout="wide"
)

apply_styles()

st.markdown(
    '<p style="font-size:2.4rem; font-weight:700; margin-bottom:4px;">'
    '<span style="color:#EA288D;">Feature Impact & Analysis</span></p>',
    unsafe_allow_html=True
)
st.markdown("Visualize which factors most influence PCOS diagnosis")

@st.cache_data
def load_and_analyze():
    possible_paths = [
        Path(__file__).parent.parent.parent / 'data' / 'processed' / 'cleaned_data.csv',
        Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_data.csv',
        Path('data') / 'processed' / 'cleaned_data.csv'
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(f"Could not find cleaned_data.csv. Tried: {possible_paths}")

    df = pd.read_csv(data_path)

    numerical_cols = ['age_yrs', 'weight_kg', 'heightcm', 'bmi', 'pulse_ratebpm', 'rr_breaths_min',
                      'hbg_dl', 'cycle_lengthdays', 'fsh_miu_ml', 'lh_miu_ml', 'fsh_lh',
                      'hipinch', 'waistinch', 'waist_hip_ratio', 'tsh_miu_l', 'amhng_ml',
                      'prlng_ml', 'vit_d3_ng_ml', 'prgng_ml', 'rbsmg_dl', 'bp_systolic_mmhg',
                      'bp_diastolic_mmhg', 'follicle_no_l', 'follicle_no_r',
                      'avg_f_size_l_mm', 'avg_f_size_r_mm', 'endometrium_mm',
                      'blood_group', 'marraige_status_yrs', 'no_of_aborptions',
                      'i_beta_hcg_miu_ml', 'ii_beta_hcg_miu_ml']

    binary_cols = ['cycle_r_i', 'pregnant_y_n', 'weight_gain_y_n', 'hair_growth_y_n',
                   'skin_darkening_y_n', 'hair_loss_y_n', 'pimples_y_n',
                   'fast_food_y_n', 'reg_exercise_y_n']

    all_feature_cols = numerical_cols + binary_cols

    # ii_beta_hcg_miu_ml can have mixed types in source data
    df['ii_beta_hcg_miu_ml'] = pd.to_numeric(df['ii_beta_hcg_miu_ml'], errors='coerce')

    X = df[all_feature_cols].fillna(df[all_feature_cols].median(numeric_only=True))
    y = df['pcos_y_n']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

    feature_importance = pd.DataFrame({
        'Feature': all_feature_cols,
        'Model Coefficient': model.coef_[0],
        'MI Score': mi_scores,
        'Abs Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs Coefficient', ascending=False)

    return df, X, y, all_feature_cols, feature_importance, model


def run_model_comparison():
    # Precomputed via 5-fold stratified CV with fixed splits across all models (NB07).
    # Full feature set: 41 features. Low-cost: 18 non-invasive features.
    # All pairwise differences non-significant (Wilcoxon signed-rank, Bonferroni-corrected).
    results = [
        {'Feature Set': 'Full (41 features)',          'Algorithm': 'Logistic Regression', 'ROC-AUC': 0.9419, 'ROC-AUC SD': 0.0026},
        {'Feature Set': 'Full (41 features)',          'Algorithm': 'Ridge',               'ROC-AUC': 0.9505, 'ROC-AUC SD': 0.0035},
        {'Feature Set': 'Full (41 features)',          'Algorithm': 'LASSO',               'ROC-AUC': 0.9551, 'ROC-AUC SD': 0.0093},
        {'Feature Set': 'Full (41 features)',          'Algorithm': 'XGBoost',             'ROC-AUC': 0.9623, 'ROC-AUC SD': 0.0123},
        {'Feature Set': 'Full (41 features)',          'Algorithm': 'Random Forest',       'ROC-AUC': 0.9610, 'ROC-AUC SD': 0.0119},
        {'Feature Set': 'Non-Invasive (18 features)', 'Algorithm': 'Logistic Regression', 'ROC-AUC': 0.8778, 'ROC-AUC SD': 0.0271},
        {'Feature Set': 'Non-Invasive (18 features)', 'Algorithm': 'Ridge',               'ROC-AUC': 0.8803, 'ROC-AUC SD': 0.0272},
        {'Feature Set': 'Non-Invasive (18 features)', 'Algorithm': 'LASSO',               'ROC-AUC': 0.8853, 'ROC-AUC SD': 0.0237},
        {'Feature Set': 'Non-Invasive (18 features)', 'Algorithm': 'XGBoost',             'ROC-AUC': 0.8750, 'ROC-AUC SD': 0.0183},
        {'Feature Set': 'Non-Invasive (18 features)', 'Algorithm': 'Random Forest',       'ROC-AUC': 0.8891, 'ROC-AUC SD': 0.0242},
    ]
    return pd.DataFrame(results)


try:
    df, X, y, all_feature_cols, feature_importance, model = load_and_analyze()
except FileNotFoundError:
    st.error("Data Loading Error")
    st.error("Could not find the required data file: `data/processed/cleaned_data.csv`")
    st.stop()
except Exception as e:
    st.error("Analysis Error")
    st.error(f"Failed to perform feature impact analysis: {str(e)}")
    st.stop()

st.sidebar.markdown("### Analysis Options")
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    options=['Feature Importance', 'Correlation Heatmap', 'PCOS vs Non-PCOS Distribution', 'Model Comparison'],
    index=0
)

st.divider()

if analysis_type == 'Feature Importance':
    st.markdown("### Feature Importance Ranking")
    st.markdown("*Shows which features are most predictive of PCOS diagnosis (Logistic Regression coefficients)*")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(feature_importance[['Feature', 'Abs Coefficient']].head(10),
                     use_container_width=True, hide_index=True)

    with col2:
        top_features = feature_importance.head(12)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(top_features)))
        ax.barh(range(len(top_features)), top_features['Abs Coefficient'],
                color=colors, edgecolor='black', alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Absolute Model Coefficient', fontsize=11)
        ax.set_title('Top 12 Features Most Associated with PCOS', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        style_fig(fig, ax)
        st.pyplot(fig)

    st.divider()

    st.markdown("### Mutual Information Analysis")
    st.markdown("*Measures information gain about PCOS status from each feature*")

    col1, col2 = st.columns([1, 2])

    with col1:
        mi_top = feature_importance.nlargest(10, 'MI Score')[['Feature', 'MI Score']]
        st.dataframe(mi_top, use_container_width=True, hide_index=True)

    with col2:
        mi_top_full = feature_importance.nlargest(12, 'MI Score')
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(mi_top_full)))
        ax.barh(range(len(mi_top_full)), mi_top_full['MI Score'],
                color=colors, edgecolor='black', alpha=0.8)
        ax.set_yticks(range(len(mi_top_full)))
        ax.set_yticklabels(mi_top_full['Feature'])
        ax.set_xlabel('Mutual Information Score', fontsize=11)
        ax.set_title('Top 12 Features by Information Content', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        style_fig(fig, ax)
        st.pyplot(fig)

elif analysis_type == 'Correlation Heatmap':
    st.markdown("### Feature Correlation Analysis")

    selected_features = st.multiselect(
        "Select features to include",
        all_feature_cols,
        default=['follicle_no_r', 'follicle_no_l', 'amhng_ml', 'lh_miu_ml', 'fsh_miu_ml',
                 'bmi', 'weight_kg', 'waist_hip_ratio', 'age_yrs']
    )

    if selected_features:
        corr_data = df[selected_features].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    cbar_kws={'label': 'Correlation Coefficient'}, ax=ax,
                    linewidths=0.5, square=True)
        ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        style_fig(fig, ax)
        st.pyplot(fig)

elif analysis_type == 'PCOS vs Non-PCOS Distribution':
    st.markdown("### Feature Distribution: PCOS vs Control Group")

    selected_features = st.multiselect(
        "Select features to compare",
        all_feature_cols,
        default=['follicle_no_r', 'follicle_no_l', 'amhng_ml', 'bmi', 'lh_miu_ml']
    )

    if selected_features:
        n_features = len(selected_features)
        n_cols = 2
        n_rows = (n_features + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(selected_features):
            ax = axes[idx]
            pcos_data = df[df['pcos_y_n'] == 1][feature]
            control_data = df[df['pcos_y_n'] == 0][feature]

            parts = ax.violinplot([control_data.dropna(), pcos_data.dropna()],
                                  positions=[0, 1], showmeans=True, showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Control', 'PCOS'])
            ax.set_ylabel(feature, fontsize=10)
            ax.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            mean_control = control_data.mean()
            mean_pcos = pcos_data.mean()
            pct_diff = ((mean_pcos - mean_control) / mean_control * 100) if mean_control != 0 else 0
            ax.text(0.5, 0.95, f'Mean diff: {pct_diff:+.1f}%',
                    transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        for idx in range(len(selected_features), len(axes)):
            axes[idx].set_visible(False)

        style_fig(fig, axes)
        st.pyplot(fig)

elif analysis_type == 'Model Comparison':
    st.markdown("### Model Comparison: All Five Models")
    st.markdown(
        "5-fold stratified CV with identical splits across all models. "
        "Error bars show ± 1 SD. All pairwise differences are non-significant."
    )

    comparison_df = run_model_comparison()

    display_df = comparison_df.copy()
    display_df['AUC-ROC (mean ± SD)'] = display_df.apply(
        lambda r: f"{r['ROC-AUC']:.4f} ± {r['ROC-AUC SD']:.3f}", axis=1
    )
    st.dataframe(
        display_df[['Feature Set', 'Algorithm', 'AUC-ROC (mean ± SD)']],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    PALETTE = ['#4F86C6', '#F4A261', '#2a9d8f', '#E63946', '#8338ec']
    feature_sets = comparison_df['Feature Set'].unique()
    algorithms = comparison_df['Algorithm'].unique()
    x = np.arange(len(feature_sets))
    width = 0.15
    algo_colors = dict(zip(algorithms, PALETTE))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, algo in enumerate(algorithms):
        subset = comparison_df[comparison_df['Algorithm'] == algo]
        vals = [subset[subset['Feature Set'] == fs]['ROC-AUC'].values[0] for fs in feature_sets]
        errs = [subset[subset['Feature Set'] == fs]['ROC-AUC SD'].values[0] for fs in feature_sets]
        offset = (i - len(algorithms) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=algo, color=algo_colors[algo],
               alpha=0.85, edgecolor='black', yerr=errs, capsize=4)

    ax.set_title('5-Fold CV AUC-ROC — All Five Models', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_sets, fontsize=10)
    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel('AUC-ROC')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=9)
    style_fig(fig, ax)
    st.pyplot(fig)

    st.divider()
    st.markdown("""
    **Notes:**
    - **Full model** uses all 41 clinical features
    - **Non-invasive model** uses 18 features: vitals, symptoms, cycle pattern, and lifestyle — no blood tests or ultrasound
    - Cross-validation uses 5 stratified folds with the same random splits for all models to enable paired comparisons
    - All pairwise AUC differences are non-significant after Bonferroni correction (10 comparisons per feature set)
    """)

st.divider()

st.markdown("### Dataset Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Features", len(all_feature_cols))
    st.metric("PCOS Cases", (y == 1).sum())

with col2:
    st.metric("Control Cases", (y == 0).sum())
    st.metric("PCOS Prevalence", f"{(y == 1).sum() / len(y) * 100:.1f}%")

with col3:
    st.metric("Feature Range", f"Min: {X.min().min():.1f}")
    st.metric("Max Value", f"Max: {X.max().max():.1f}")
