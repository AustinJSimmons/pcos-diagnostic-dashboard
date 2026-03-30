import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
import sys
sys.path.append(str(Path(__file__).parent.parent))
from styles import apply_styles, style_fig

st.set_page_config(
    page_title="Phenotype Explorer",
    layout="wide"
)

apply_styles()

st.markdown(
    '<p style="font-size:2.4rem; font-weight:700; margin-bottom:4px;">'
    '<span style="color:#EA288D;">PCOS Phenotype Explorer</span></p>',
    unsafe_allow_html=True
)
st.markdown("Discover distinct PCOS phenotypes and their clinical characteristics")

# Load data
@st.cache_data
def load_data():
    # Try multiple path resolutions for robustness
    possible_paths = [
        Path(__file__).parent.parent.parent / 'data' / 'processed' / 'cleaned_data.csv',  # From pages/
        Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_data.csv',  # From app/
        Path('data') / 'processed' / 'cleaned_data.csv'  # From project root
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            return pd.read_csv(data_path)
    
    raise FileNotFoundError(f"Could not find cleaned_data.csv. Tried: {possible_paths}")

@st.cache_data
def perform_clustering(df):
    """Perform K-Means clustering on PCOS patients"""
    # Select numerical features
    numerical_cols = ['age_yrs', 'weight_kg', 'heightcm', 'bmi', 'pulse_ratebpm', 'rr_breaths_min',
                      'hbg_dl', 'cycle_lengthdays', 'fsh_miu_ml', 'lh_miu_ml', 'fsh_lh',
                      'hipinch', 'waistinch', 'waist_hip_ratio', 'tsh_miu_l', 'amhng_ml',
                      'prlng_ml', 'vit_d3_ng_ml', 'prgng_ml', 'rbsmg_dl', 'bp_systolic_mmhg',
                      'bp_diastolic_mmhg', 'follicle_no_l', 'follicle_no_r',
                      'avg_f_size_l_mm', 'avg_f_size_r_mm', 'endometrium_mm']
    
    # Filter PCOS patients only
    df_pcos = df[df['pcos_y_n'] == 1].reset_index(drop=True)
    
    # Prepare and scale data
    X_pcos = df_pcos[numerical_cols].fillna(df_pcos[numerical_cols].mean())
    scaler = StandardScaler()
    X_pcos_scaled = scaler.fit_transform(X_pcos)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pcos_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_pcos_scaled)
    
    return df_pcos, clusters, X_pca, pca, kmeans, X_pcos_scaled, numerical_cols

@st.cache_data
def identify_differentiating_features(df_pcos, clusters, numerical_cols, top_n=10):
    """Identify which features most differentiate the two phenotypes"""
    pheno0 = df_pcos[clusters == 0][numerical_cols].fillna(df_pcos[numerical_cols].mean())
    pheno1 = df_pcos[clusters == 1][numerical_cols].fillna(df_pcos[numerical_cols].mean())
    
    # Calculate effect sizes (Cohen's d) for each feature
    effect_sizes = []
    for col in numerical_cols:
        mean_diff = pheno1[col].mean() - pheno0[col].mean()
        pooled_std = np.sqrt(((len(pheno0)-1)*pheno0[col].std()**2 + (len(pheno1)-1)*pheno1[col].std()**2) / 
                            (len(pheno0) + len(pheno1) - 2))
        if pooled_std > 0:
            cohens_d = mean_diff / pooled_std
        else:
            cohens_d = 0
        
        # T-test p-value
        t_stat, p_val = ttest_ind(pheno0[col].dropna(), pheno1[col].dropna())
        
        effect_sizes.append({
            'feature': col,
            'pheno0_mean': pheno0[col].mean(),
            'pheno1_mean': pheno1[col].mean(),
            'cohens_d': cohens_d,
            'p_value': p_val,
            'pct_diff': abs(mean_diff) / (abs(pheno0[col].mean()) + 0.001) * 100
        })
    
    effect_df = pd.DataFrame(effect_sizes).sort_values('cohens_d', key=abs, ascending=False)
    return effect_df, pheno0, pheno1

try:
    df = load_data()
    df_pcos, clusters, X_pca, pca, kmeans, X_pcos_scaled, numerical_cols = perform_clustering(df)
    effect_df, pheno0, pheno1 = identify_differentiating_features(df_pcos, clusters, numerical_cols, top_n=10)

    # Dynamically assign phenotype names based on BMI (higher BMI = Metabolic)
    bmi_0 = df_pcos[clusters == 0]['bmi'].mean()
    bmi_1 = df_pcos[clusters == 1]['bmi'].mean()
    if bmi_0 > bmi_1:
        pheno_names = {0: "Metabolic PCOS", 1: "Lean PCOS"}
        pheno_desc  = {0: "Higher BMI, weight & metabolic markers",
                       1: "Lower BMI, leaner profile with similar follicle counts"}
    else:
        pheno_names = {0: "Lean PCOS", 1: "Metabolic PCOS"}
        pheno_desc  = {0: "Lower BMI, leaner profile with similar follicle counts",
                       1: "Higher BMI, weight & metabolic markers"}
    pheno_colors_map = {0: '#EA288D', 1: '#950F54'}

    # Sidebar controls
    st.sidebar.markdown("### Visualization Controls")
    view_mode = st.sidebar.radio("Select View", ["Overview", "Detailed Comparison", "Feature Deep-Dive", "Classify Patient"])
    pheno_count_0 = (clusters == 0).sum()
    pheno_count_1 = (clusters == 1).sum()
    
    if view_mode == "Overview":
        # Overview Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Phenotype Clustering (PCA Projection)")
            st.markdown("*Click below to explore how PCOS patients naturally group by clinical features*")
            
            # Create scatter plot with better styling
            fig, ax = plt.subplots(figsize=(10, 7))
            pheno_colors = ['#EA288D', '#950F54']
            
            for pheno in [0, 1]:
                mask = clusters == pheno
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=pheno_colors_map[pheno], alpha=0.6, s=120,
                          label=f'{pheno_names[pheno]} (n={mask.sum()})',
                          edgecolors='white', linewidth=0.5)

            # Plot centroids — styled diamonds
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            for i, (cx, cy) in enumerate(centroids_pca):
                ax.scatter(cx, cy, c=pheno_colors_map[i], marker='D',
                          s=300, edgecolors='white', linewidth=2, zorder=5)
                ax.scatter(cx, cy, c='white', marker='D',
                          s=80, zorder=6)
            # Single neutral legend entry for centroids
            ax.scatter([], [], c='#444', marker='D', s=100,
                      edgecolors='white', linewidth=1.5, label='Cluster Centroid')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, fontweight='bold')
            ax.set_title('K-Means Clustering of PCOS Patients', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=10, framealpha=0.95)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.markdown("### Population Distribution")

            phenotype_counts = [pheno_count_0, pheno_count_1]
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = ['#EA288D', '#950F54']
            wedges, texts, autotexts = ax.pie(phenotype_counts,
                                              labels=[f'{pheno_names[0]}\n(n={phenotype_counts[0]})',
                                                      f'{pheno_names[1]}\n(n={phenotype_counts[1]})'],
                                              autopct='%1.1f%%', colors=colors, startangle=90,
                                              textprops={'fontsize': 11, 'weight': 'bold'})

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_weight('bold')

            ax.set_title('PCOS Distribution', fontsize=12, fontweight='bold', pad=15)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)
        
        # Phenotype characteristics cards
        st.divider()
        st.markdown("### Phenotype Characteristics at a Glance")
        
        # Top differentiating features
        top_diff_features = effect_df.head(6)['feature'].tolist()
        
        col1, col2 = st.columns(2)
        
        for pheno_idx, col in zip([0, 1], [col1, col2]):
            pheno_data = df_pcos[clusters == pheno_idx][numerical_cols]
            
            with col:
                st.markdown(f"#### **{pheno_names[pheno_idx]}** (n={sum(clusters == pheno_idx)} patients)")
                st.caption(pheno_desc[pheno_idx])
                
                # Key statistics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Avg Age", f"{pheno_data['age_yrs'].mean():.1f} yrs")
                with metrics_col2:
                    st.metric("Avg BMI", f"{pheno_data['bmi'].mean():.1f} kg/m²")
                with metrics_col3:
                    st.metric("Avg Follicles", f"{(pheno_data['follicle_no_l'].mean() + pheno_data['follicle_no_r'].mean()):.0f}")
                
                # Key characteristics
                st.markdown("**Key Characteristics:**")
                char_text = f"""
                - **Ovarian Features**: {pheno_data['follicle_no_r'].mean():.0f} right, {pheno_data['follicle_no_l'].mean():.0f} left follicles
                - **Hormonal**: LH {pheno_data['lh_miu_ml'].mean():.2f}, FSH {pheno_data['fsh_miu_ml'].mean():.2f} mIU/ml
                - **Metabolic**: Weight {pheno_data['weight_kg'].mean():.1f} kg, Waist-Hip {pheno_data['waist_hip_ratio'].mean():.2f}
                - **Reserve**: AMH {pheno_data['amhng_ml'].mean():.2f} ng/ml
                """
                st.markdown(char_text)
    
    elif view_mode == "Detailed Comparison":
        st.markdown("### Detailed Phenotype Comparison")
        st.markdown("*Click a feature to highlight the key differences between phenotypes*")
        
        # Top differentiating features table
        comparison_features = effect_df.head(10).copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Top 10 Differentiating Features")
            
            # Create comparison visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x_pos = np.arange(len(comparison_features))
            effect_values = comparison_features['cohens_d'].values
            colors_effects = ['#EA288D' if x > 0 else '#950F54' for x in effect_values]
            
            bars = ax.barh(x_pos, effect_values, color=colors_effects, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax.set_yticks(x_pos)
            ax.set_yticklabels(comparison_features['feature'].values, fontsize=11)
            ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12, fontweight='bold')
            ax.set_title(f"Differentiating Features Between Phenotypes\n"
                        f"(Pink = Higher in {pheno_names[1]}, Purple = Higher in {pheno_names[0]})",
                        fontsize=12, fontweight='bold', pad=15)
            ax.axvline(x=0, color='#9E9E9E', linestyle='-', linewidth=1.2)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            for i, (idx, row) in enumerate(comparison_features.iterrows()):
                pct = row['pct_diff']
                ax.text(row['cohens_d'], i, f"  {pct:.0f}%",
                       va='center', fontsize=9, fontweight='bold')

            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Statistical Significance")
            sig_features = comparison_features[comparison_features['p_value'] < 0.05]
            st.metric("Statistically Significant", f"{len(sig_features)}/10", "at p<0.05")
            st.markdown("**Features with largest effect:**")
            for idx, row in comparison_features.head(5).iterrows():
                st.markdown(f"- **{row['feature']}**: d={row['cohens_d']:.2f}")
        
        st.divider()
        
        # Detailed statistics table
        st.markdown("### 📋 Feature-by-Feature Comparison")
        
        # Create comprehensive comparison table
        comparison_table = []
        for feature in comparison_features['feature'].head(10):
            comparison_table.append({
                'Feature': feature,
                'Phenotype 0 Mean': f"{effect_df[effect_df['feature']==feature]['pheno0_mean'].values[0]:.2f}",
                'Phenotype 1 Mean': f"{effect_df[effect_df['feature']==feature]['pheno1_mean'].values[0]:.2f}",
                'Difference': f"{abs(effect_df[effect_df['feature']==feature]['pheno1_mean'].values[0] - effect_df[effect_df['feature']==feature]['pheno0_mean'].values[0]):.2f}",
                'Effect Size': f"{effect_df[effect_df['feature']==feature]['cohens_d'].values[0]:.2f}",
                'P-Value': f"{effect_df[effect_df['feature']==feature]['p_value'].values[0]:.4f}"
            })
        
        comparison_table_df = pd.DataFrame(comparison_table)
        st.dataframe(comparison_table_df, use_container_width=True, hide_index=True)
    
    elif view_mode == "Feature Deep-Dive":
        st.markdown("### Feature Deep-Dive Analysis")
        
        # Feature selector
        selected_feature = st.selectbox("Select Feature to Compare",
                                       options=numerical_cols,
                                       index=0)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### {selected_feature} Distribution by Phenotype")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            data_p0 = df_pcos[df_pcos.index.isin(np.where(clusters == 0)[0])][selected_feature].dropna()
            data_p1 = df_pcos[df_pcos.index.isin(np.where(clusters == 1)[0])][selected_feature].dropna()
            
            parts = ax.violinplot([data_p0, data_p1], positions=[0, 1], 
                                 showmeans=True, showmedians=True, widths=0.7)
            
            # Customize violin plot colors
            for pc in parts['bodies']:
                pc.set_facecolor('#EA288D')
                pc.set_alpha(0.6)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Phenotype 0', 'Phenotype 1'], fontsize=11, fontweight='bold')
            ax.set_ylabel(selected_feature, fontsize=12, fontweight='bold')
            ax.set_title(f'{selected_feature} Distribution Comparison', fontsize=13, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)
        
        with col1:
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown("**Phenotype 0**")
                st.metric("Mean", f"{data_p0.mean():.2f}")
                st.metric("Std Dev", f"{data_p0.std():.2f}")
                st.metric("Median", f"{data_p0.median():.2f}")
            
            with col_stat2:
                st.markdown("**Phenotype 1**")
                st.metric("Mean", f"{data_p1.mean():.2f}")
                st.metric("Std Dev", f"{data_p1.std():.2f}")
                st.metric("Median", f"{data_p1.median():.2f}")
        
        with col2:
            st.markdown("**Statistical Test**")
            t_stat, p_val = ttest_ind(data_p0, data_p1)
            
            # Find effect size
            effect = effect_df[effect_df['feature'] == selected_feature]
            if not effect.empty:
                cohens_d = effect['cohens_d'].values[0]
                st.metric("Cohen's d", f"{cohens_d:.3f}")
            
            st.metric("P-Value", f"{p_val:.4f}")
            
            if p_val < 0.05:
                st.success("Significantly Different")
            else:
                st.info("No significant difference")
            
            mean_diff_pct = abs(data_p1.mean() - data_p0.mean()) / data_p0.mean() * 100
            st.metric("% Difference", f"{mean_diff_pct:.1f}%")

    elif view_mode == "Classify Patient":
        st.markdown("### Classify a New Patient")
        st.markdown("Enter a patient's clinical values to see which PCOS phenotype they most resemble.")

        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            c_age   = st.number_input("Age (yrs)", 15, 50, 28, key="c_age")
            c_bmi   = st.number_input("BMI (kg/m²)", 15.0, 45.0, 25.0, key="c_bmi")
            c_wt    = st.number_input("Weight (kg)", 35.0, 120.0, 65.0, key="c_wt")
            c_ht    = st.number_input("Height (cm)", 140.0, 185.0, 160.0, key="c_ht")
            c_pulse = st.number_input("Pulse (bpm)", 40, 120, 75, key="c_pulse")
            c_rr    = st.number_input("Resp. Rate (br/min)", 10, 40, 16, key="c_rr")

        with col2:
            c_hb    = st.number_input("Haemoglobin (g/dL)", 5.0, 18.0, 13.0, key="c_hb")
            c_cycle = st.number_input("Cycle Length (days)", 15, 90, 30, key="c_cycle")
            c_fsh   = st.number_input("FSH (mIU/ml)", 0.5, 20.0, 6.0, key="c_fsh")
            c_lh    = st.number_input("LH (mIU/ml)", 0.1, 100.0, 5.0, key="c_lh")
            c_fsh_lh= st.number_input("FSH/LH ratio", 0.1, 10.0, 1.2, key="c_fshlh")
            c_hip   = st.number_input("Hip (inch)", 28.0, 55.0, 38.0, key="c_hip")

        with col3:
            c_waist = st.number_input("Waist (inch)", 22.0, 50.0, 30.0, key="c_waist")
            c_whr   = st.number_input("Waist-Hip Ratio", 0.6, 1.1, 0.8, key="c_whr")
            c_tsh   = st.number_input("TSH (mIU/L)", 0.1, 10.0, 2.5, key="c_tsh")
            c_amh   = st.number_input("AMH (ng/ml)", 0.0, 15.0, 5.0, key="c_amh")
            c_prl   = st.number_input("Prolactin (ng/ml)", 0.0, 100.0, 15.0, key="c_prl")
            c_vitd  = st.number_input("Vit D3 (ng/ml)", 0.0, 80.0, 20.0, key="c_vitd")

        # Additional fields in expander
        with st.expander("More features (optional — uses dataset mean if left at default)"):
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                c_prg  = st.number_input("Progesterone (ng/ml)", 0.0, 10.0, 0.5, key="c_prg")
                c_rbs  = st.number_input("Random Blood Sugar (mg/dL)", 50.0, 300.0, 90.0, key="c_rbs")
                c_bps  = st.number_input("Systolic BP (mmHg)", 80, 180, 120, key="c_bps")
                c_bpd  = st.number_input("Diastolic BP (mmHg)", 50, 120, 80, key="c_bpd")
            with ex_col2:
                c_fl   = st.number_input("Follicles Left", 0, 30, 10, key="c_fl")
                c_fr   = st.number_input("Follicles Right", 0, 30, 10, key="c_fr")
                c_afl  = st.number_input("Avg Follicle Size L (mm)", 0.0, 30.0, 15.0, key="c_afl")
                c_afr  = st.number_input("Avg Follicle Size R (mm)", 0.0, 30.0, 15.0, key="c_afr")
            c_endo = st.number_input("Endometrium (mm)", 0.0, 20.0, 8.0, key="c_endo")

        if st.button("Classify Patient", use_container_width=True):
            patient_vals = [c_age, c_wt, c_ht, c_bmi, c_pulse, c_rr, c_hb, c_cycle,
                            c_fsh, c_lh, c_fsh_lh, c_hip, c_waist, c_whr, c_tsh,
                            c_amh, c_prl, c_vitd, c_prg, c_rbs, c_bps, c_bpd,
                            c_fl, c_fr, c_afl, c_afr, c_endo]

            from sklearn.preprocessing import StandardScaler as _SS
            # Use the same scaler fitted on the PCOS cluster data
            patient_arr = np.array([patient_vals])
            # Re-fit scaler on full PCOS data (same as perform_clustering does)
            X_pcos_raw = df_pcos[numerical_cols].fillna(df_pcos[numerical_cols].mean())
            scaler_cls = _SS()
            scaler_cls.fit(X_pcos_raw)
            patient_scaled = scaler_cls.transform(patient_arr)

            predicted_cluster = kmeans.predict(patient_scaled)[0]
            patient_pca = pca.transform(patient_scaled)

            # Distances to both centroids
            dists = np.linalg.norm(kmeans.cluster_centers_ - patient_scaled, axis=1)
            confidence = 1 - (dists[predicted_cluster] / dists.sum())

            st.divider()
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                color = pheno_colors_map[predicted_cluster]
                st.markdown(
                    f'<div style="background:{color}22; border-left:5px solid {color}; '
                    f'border-radius:12px; padding:20px; margin-bottom:12px;">'
                    f'<p style="font-size:1.05rem; font-weight:600; color:{color}; margin:0;">'
                    f'Predicted Phenotype</p>'
                    f'<p style="font-size:1.6rem; font-weight:700; color:#1C1C2E; margin:4px 0;">'
                    f'{pheno_names[predicted_cluster]}</p>'
                    f'<p style="color:#666; font-size:0.9rem; margin:0;">'
                    f'{pheno_desc[predicted_cluster]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.metric("Cluster Confidence", f"{confidence*100:.1f}%")
                st.metric("Distance to Centroid", f"{dists[predicted_cluster]:.2f}")

            with res_col2:
                fig, ax = plt.subplots(figsize=(9, 6))
                for pheno in [0, 1]:
                    mask = clusters == pheno
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                              c=pheno_colors_map[pheno], alpha=0.35, s=80,
                              label=pheno_names[pheno], edgecolors='none')

                # Centroids
                centroids_pca = pca.transform(kmeans.cluster_centers_)
                for i, (cx, cy) in enumerate(centroids_pca):
                    ax.scatter(cx, cy, c=pheno_colors_map[i], marker='D',
                              s=200, edgecolors='white', linewidth=1.5, zorder=5)

                # Patient point
                ax.scatter(patient_pca[0, 0], patient_pca[0, 1],
                          c='gold', marker='*', s=600,
                          edgecolors='#EA288D', linewidth=2, zorder=10,
                          label='This Patient')

                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
                ax.set_title('Patient Position in Phenotype Space', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
                style_fig(fig, ax)
                st.pyplot(fig, use_container_width=True)

except Exception as e:
    st.error("Error Loading Phenotype Data")
    st.error(f"Could not load or process clustering data: {str(e)}")
    st.info("Make sure the cleaned data file exists at: `data/processed/cleaned_data.csv`")
