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
                      'avg_f_size_l_mm', 'avg_f_size_r_mm', 'endometrium_mm']

    X = df[numerical_cols].fillna(df[numerical_cols].mean())
    y = df['pcos_y_n']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

    feature_importance = pd.DataFrame({
        'Feature': numerical_cols,
        'Model Coefficient': model.coef_[0],
        'MI Score': mi_scores,
        'Abs Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs Coefficient', ascending=False)

    return df, X, y, numerical_cols, feature_importance, model


def run_model_comparison():
    # Precomputed via 5-fold stratified CV using tuned hyperparameters from notebook
    # (03-xgboost-shap.ipynb grid search). Pipeline used to prevent data leakage.
    results = [
        {'Feature Set': 'Full (9 features)',          'Algorithm': 'Logistic Regression', 'Accuracy': 0.8503, 'Accuracy SD': 0.0289, 'ROC-AUC': 0.9051, 'ROC-AUC SD': 0.0018, 'F1 Score': 0.7595, 'F1 SD': 0.0468},
        {'Feature Set': 'Full (9 features)',          'Algorithm': 'XGBoost',             'Accuracy': 0.8539, 'Accuracy SD': 0.0298, 'ROC-AUC': 0.9094, 'ROC-AUC SD': 0.0211, 'F1 Score': 0.7775, 'F1 SD': 0.0531},
        {'Feature Set': 'Non-Invasive (17 features)', 'Algorithm': 'Logistic Regression', 'Accuracy': 0.8447, 'Accuracy SD': 0.0355, 'ROC-AUC': 0.8662, 'ROC-AUC SD': 0.0307, 'F1 Score': 0.7536, 'F1 SD': 0.0516},
        {'Feature Set': 'Non-Invasive (17 features)', 'Algorithm': 'XGBoost',             'Accuracy': 0.8170, 'Accuracy SD': 0.0358, 'ROC-AUC': 0.8744, 'ROC-AUC SD': 0.0278, 'F1 Score': 0.7220, 'F1 SD': 0.0499},
    ]
    return pd.DataFrame(results)


try:
    df, X, y, numerical_cols, feature_importance, model = load_and_analyze()
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
        numerical_cols,
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
        numerical_cols,
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
    st.markdown("### Model Comparison: Logistic Regression vs XGBoost")
    st.markdown(
        "5-fold cross-validated performance across both feature sets. "
        "Error bars show ± 1 standard deviation across folds."
    )

    with st.spinner("Running cross-validation (this may take a moment)..."):
        comparison_df = run_model_comparison()

    # Summary table — SD columns: 'Accuracy SD', 'ROC-AUC SD', 'F1 SD'
    sd_col_map = {'Accuracy': 'Accuracy SD', 'ROC-AUC': 'ROC-AUC SD', 'F1 Score': 'F1 SD'}
    display_df = comparison_df.copy()
    for metric, sd_col in sd_col_map.items():
        display_df[metric] = display_df.apply(
            lambda r, m=metric, s=sd_col: f"{r[m]:.3f} ± {r[s]:.3f}", axis=1
        )
    st.dataframe(
        display_df[['Feature Set', 'Algorithm', 'Accuracy', 'ROC-AUC', 'F1 Score']],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Bar charts for each metric
    metrics = ['Accuracy', 'ROC-AUC', 'F1 Score']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    feature_sets = comparison_df['Feature Set'].unique()
    algorithms = comparison_df['Algorithm'].unique()
    x = np.arange(len(feature_sets))
    width = 0.35
    algo_colors = {'Logistic Regression': '#EA288D', 'XGBoost': '#950F54'}

    for ax, metric in zip(axes, metrics):
        for i, algo in enumerate(algorithms):
            subset = comparison_df[comparison_df['Algorithm'] == algo]
            vals = [subset[subset['Feature Set'] == fs][metric].values[0] for fs in feature_sets]
            errs = [subset[subset['Feature Set'] == fs][sd_col_map[metric]].values[0] for fs in feature_sets]
            bars = ax.bar(x + i * width - width / 2, vals, width,
                          label=algo, color=algo_colors[algo], alpha=0.85,
                          edgecolor='black', yerr=errs, capsize=5)

        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=9)

        # Value labels on bars
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Logistic Regression vs XGBoost — 5-Fold CV Performance',
                 fontsize=14, fontweight='bold', y=1.02)
    style_fig(fig, axes)
    st.pyplot(fig)

    st.divider()
    st.markdown("""
    **Notes:**
    - **Full model** uses 9 features: age, BMI, weight, waist-hip ratio, follicle counts (L/R), AMH, LH, FSH
    - **Non-invasive model** uses 17 features: vitals, symptoms, and lifestyle factors only — no blood tests or ultrasound
    - Cross-validation uses 5 stratified folds to preserve class balance across splits
    - XGBoost uses `scale_pos_weight` to handle the ~2:1 class imbalance (controls vs PCOS)
    """)

st.divider()

st.markdown("### Dataset Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Features", len(numerical_cols))
    st.metric("PCOS Cases", (y == 1).sum())

with col2:
    st.metric("Control Cases", (y == 0).sum())
    st.metric("PCOS Prevalence", f"{(y == 1).sum() / len(y) * 100:.1f}%")

with col3:
    st.metric("Feature Range", f"Min: {X.min().min():.1f}")
    st.metric("Max Value", f"Max: {X.max().max():.1f}")
