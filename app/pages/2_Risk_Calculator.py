import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))
from styles import apply_styles, style_fig

st.set_page_config(
    page_title="PCOS Risk Calculator",
    layout="wide"
)

apply_styles()

st.markdown(
    '<p style="font-size:2.4rem; font-weight:700; margin-bottom:4px;">'
    '<span style="color:#EA288D;">PCOS Risk Screening Calculator</span></p>',
    unsafe_allow_html=True
)
st.markdown("Calculate personalized PCOS risk based on clinical indicators")

@st.cache_data
def train_models():
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

    full_features = ['age_yrs', 'bmi', 'follicle_no_r', 'follicle_no_l', 'amhng_ml',
                     'lh_miu_ml', 'fsh_miu_ml', 'weight_kg', 'waist_hip_ratio']

    noninvasive_features = ['age_yrs', 'bmi', 'weight_kg', 'waist_hip_ratio',
                            'pulse_ratebpm', 'rr_breaths_min', 'cycle_r_i', 'cycle_lengthdays',
                            'bp_systolic_mmhg', 'bp_diastolic_mmhg', 'weight_gain_y_n',
                            'hair_growth_y_n', 'skin_darkening_y_n', 'hair_loss_y_n',
                            'pimples_y_n', 'fast_food_y_n', 'reg_exercise_y_n', 'pregnant_y_n']

    y = df['pcos_y_n']

    models = {}

    for feature_set_name, features in [('full', full_features), ('noninvasive', noninvasive_features)]:
        X = df[features].fillna(df[features].median(numeric_only=True))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logreg = LogisticRegression(random_state=42, max_iter=1000)
        logreg.fit(X_scaled, y)

        # same params as NB07 so results are comparable
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(X_scaled, y)

        models[feature_set_name] = {
            'logreg': {'model': logreg, 'scaler': scaler, 'features': features},
            'rf': {'model': rf, 'scaler': scaler, 'features': features},
            'stats': df[features].describe()
        }

    return models

try:
    models_dict = train_models()
except FileNotFoundError:
    st.error("Data Loading Error")
    st.error("Could not find the required data file: `data/processed/cleaned_data.csv`")
    st.stop()
except Exception as e:
    st.error("Model Training Error")
    st.error(f"Failed to train the risk assessment models: {str(e)}")
    st.stop()

st.markdown("""
### How to Use
1. **Select a feature set** — Full (includes labs & ultrasound) or Non-Invasive (symptoms & vitals only)
2. **Select an algorithm** — Logistic Regression (interpretable, well-calibrated) or Random Forest (highest sensitivity on non-invasive set)
3. **Enter patient values** in the input fields below
4. **View risk assessment** and contributing factors
""")

st.divider()

# model selection
st.markdown("### Model Selection")

sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 2])

with sel_col1:
    feature_set = st.radio("Feature Set", options=['Full Model', 'Non-Invasive Model'])

with sel_col2:
    algorithm = st.radio("Algorithm", options=['Logistic Regression', 'Random Forest'])

with sel_col3:
    if feature_set == 'Full Model':
        st.info("""
        **Full Model** includes comprehensive clinical data:
        - Clinical measurements (age, BMI, weight, waist-hip ratio)
        - Blood hormone levels (LH, FSH, AMH)
        - Ultrasound findings (follicle counts)
        """)
    else:
        st.info("""
        **Non-Invasive Model** uses only accessible clinical data:
        - Demographics and body measurements
        - Vital signs (pulse, respiration, blood pressure)
        - Menstrual history (cycle regularity & length), PCOS symptoms, and lifestyle factors
        - Best for screening without blood tests or ultrasound
        """)

st.divider()

# pick the right model dict based on selections above
feature_key = 'full' if feature_set == 'Full Model' else 'noninvasive'
algo_key = 'logreg' if algorithm == 'Logistic Regression' else 'rf'
model_info = models_dict[feature_key][algo_key]
model_type_label = f"{feature_set} — {algorithm}"

# patient input fields
st.markdown("### Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (years)", min_value=15, max_value=80, value=30)
    bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=45.0, value=24.0)
    weight = st.number_input("Weight (kg)", min_value=35.0, max_value=120.0, value=65.0)
    whr = st.number_input("Waist-Hip Ratio", min_value=0.7, max_value=1.1, value=0.9)

with col2:
    pulse = st.number_input("Pulse Rate (bpm)", min_value=40, max_value=120, value=75)
    rr = st.number_input("Respiration Rate (breaths/min)", min_value=10, max_value=40, value=16)
    cycle = st.number_input("Cycle Length (days)", min_value=0, max_value=20, value=5)
    if feature_set == 'Non-Invasive Model':
        cycle_irreg = st.checkbox("Irregular Cycles", value=False,
                                  help="Irregular if cycle length value is outside 2–5; regular (2–5) reflects a normal menstrual pattern in this dataset's encoding")
    bp_sys = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=180, value=120)

with col3:
    bp_dia = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=120, value=80)

    if feature_set == 'Full Model':
        st.markdown("**Laboratory & Ultrasound Data**")
        follicle_r = st.number_input("Right Ovarian Follicles", min_value=0, max_value=30, value=10)
        follicle_l = st.number_input("Left Ovarian Follicles", min_value=0, max_value=30, value=10)
        amh = st.number_input("AMH (ng/ml)", min_value=0.0, max_value=15.0, value=5.0)
        lh = st.number_input("LH (mIU/ml)", min_value=0.1, max_value=100.0, value=5.0)
        fsh = st.number_input("FSH (mIU/ml)", min_value=1.0, max_value=20.0, value=6.0)
        weight_gain = hair_growth = skin_darkening = hair_loss = pimples = fast_food = exercise = pregnant = None
        cycle_irreg = None
    else:
        st.markdown("**PCOS Symptoms & Lifestyle**")
        weight_gain = st.checkbox("Weight Gain", value=False)
        hair_growth = st.checkbox("Excessive Hair Growth (Hirsutism)", value=False)
        skin_darkening = st.checkbox("Skin Darkening", value=False)
        hair_loss = st.checkbox("Hair Loss", value=False)
        pimples = st.checkbox("Acne/Pimples", value=False)
        fast_food = st.checkbox("High Fast Food Intake", value=False)
        exercise = st.checkbox("Regular Exercise", value=False)
        pregnant = st.checkbox("Currently Pregnant", value=False)
        follicle_r = follicle_l = amh = lh = fsh = None

st.divider()

# run predictions when the button is clicked
if st.button("Calculate Risk Score", use_container_width=True):
    st.session_state['show_results'] = True
    st.session_state['last_model_label'] = model_type_label
    if feature_set == 'Full Model':
        input_data = np.array([[age, bmi, follicle_r, follicle_l, amh, lh, fsh, weight, whr]])
    else:
        input_data = np.array([[age, bmi, weight, whr, pulse, rr, int(cycle_irreg), cycle, bp_sys, bp_dia,
                                int(weight_gain), int(hair_growth), int(skin_darkening),
                                int(hair_loss), int(pimples), int(fast_food), int(exercise), int(pregnant)]])
    input_scaled = model_info['scaler'].transform(input_data)
    st.session_state['risk_prob'] = model_info['model'].predict_proba(input_scaled)[0][1]
    st.session_state['input_scaled'] = input_scaled[0]

    if algo_key == 'logreg':
        contribs = model_info['model'].coef_[0] * input_scaled[0]
    else:
        # RF doesn't give directional per-patient contributions; multiply feature importance by
        # the standardised input value as a rough directional proxy
        contribs = model_info['model'].feature_importances_ * input_scaled[0]
    st.session_state['patient_contribs'] = pd.DataFrame({
        'Feature': model_info['features'],
        'Contribution': contribs,
    }).sort_values('Contribution', key=abs, ascending=False)
    st.session_state['shap_values'] = None

    st.session_state.pop('shap_values', None)
    st.session_state['algo_key'] = algo_key
    st.session_state['feature_set'] = feature_set
    # keep inputs in session state so the recommendation block can read them
    st.session_state['inputs'] = dict(
        bmi=bmi, pulse=pulse, bp_sys=bp_sys, bp_dia=bp_dia, cycle=cycle,
        follicle_r=follicle_r, follicle_l=follicle_l, amh=amh, lh=lh, fsh=fsh,
        weight_gain=weight_gain, hair_growth=hair_growth, skin_darkening=skin_darkening,
        hair_loss=hair_loss, pimples=pimples, fast_food=fast_food, exercise=exercise
    )

if not st.session_state.get('show_results', False):
    st.info("Fill in patient information above and click **Calculate Risk Score** to see results.")
    st.stop()

# pull everything back out of session state
risk_prob = st.session_state['risk_prob']
model_type_label = st.session_state['last_model_label']
_algo_key = st.session_state['algo_key']
_feature_set = st.session_state['feature_set']
_inputs = st.session_state['inputs']
_patient_contribs = st.session_state['patient_contribs']

# results display
st.markdown("### Risk Assessment Results")
st.markdown(f"**Model:** {model_type_label}")

col1, col2 = st.columns(2)

with col1:
    if risk_prob < 0.20:
        risk_level = "Low Risk"
    elif risk_prob < 0.50:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    st.metric("PCOS Risk Level", risk_level)
    st.metric("Risk Probability", f"{risk_prob*100:.1f}%")

with col2:
    fig, ax = plt.subplots(figsize=(8, 6))

    # 0% on the left (angle=π), 100% on the right (angle=0) — needle moves right as risk rises
    low_angles = np.linspace(np.pi, 0.80 * np.pi, 20)          # leftmost 20% → green
    ax.fill_between(np.cos(low_angles), 0, np.sin(low_angles), color='#00AA00', alpha=0.4, label='Low (0–20%)')
    ax.plot(np.cos(low_angles), np.sin(low_angles), color='#00AA00', linewidth=3)

    med_angles = np.linspace(0.80 * np.pi, 0.50 * np.pi, 30)   # middle → orange
    ax.fill_between(np.cos(med_angles), 0, np.sin(med_angles), color='#FFAA00', alpha=0.4, label='Medium (20–50%)')
    ax.plot(np.cos(med_angles), np.sin(med_angles), color='#FFAA00', linewidth=3)

    high_angles = np.linspace(0.50 * np.pi, 0, 50)              # rightmost 50% → red
    ax.fill_between(np.cos(high_angles), 0, np.sin(high_angles), color='#FF0000', alpha=0.4, label='High (50–100%)')
    ax.plot(np.cos(high_angles), np.sin(high_angles), color='#FF0000', linewidth=3)

    # invert so the needle points left at 0% and right at 100%
    needle_angle = (1 - risk_prob) * np.pi
    ax.arrow(0, 0, np.cos(needle_angle) * 0.85, np.sin(needle_angle) * 0.85,
             head_width=0.08, head_length=0.08, fc='black', ec='black', linewidth=3, zorder=10)
    ax.add_patch(plt.Circle((0, 0), 0.08, color='black', zorder=11))
    ax.plot([-1, 1], [0, 0], 'k-', linewidth=2)
    ax.text(-0.95, -0.15, '0%', ha='center', fontsize=10, fontweight='bold')
    ax.text(0, -0.15, '50%', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.95, -0.15, '100%', ha='center', fontsize=10, fontweight='bold')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10, frameon=True)
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)

st.divider()

# which features pushed the score up or down for this patient
st.markdown("### Contributing Risk Factors")
if _algo_key == 'logreg':
    st.markdown("*How each of this patient's values contributes to their log-odds of PCOS. "
                "Pink bars push risk up; purple bars push risk down.*")
else:
    st.markdown("*Feature importance weighted by standardised patient values. "
                "Pink bars indicate features above the population mean pushing risk up; purple bars indicate features below the mean.*")

top_contribs = _patient_contribs.head(9)
bar_colors = ['#C2185B' if v > 0 else '#7B1FA2' for v in top_contribs['Contribution']]

col1, col2 = st.columns([1, 2])

with col1:
    display = top_contribs[['Feature', 'Contribution']].copy()
    display['Direction'] = display['Contribution'].apply(lambda x: '⬆ Increases risk' if x > 0 else '⬇ Reduces risk')
    display['Contribution'] = display['Contribution'].apply(lambda x: f"{x:+.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

with col2:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top_contribs['Feature'], top_contribs['Contribution'],
                   color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax.axvline(0, color='#555', linewidth=1.2, linestyle='-')
    ax.set_xlabel('Contribution to Risk Score', fontsize=11)
    ax.set_title(f'Patient-Specific Risk Contributions — {model_type_label}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    style_fig(fig, ax)
    st.pyplot(fig)

st.divider()

# clinical recommendations based on what the patient entered
st.markdown("### Clinical Recommendations")

recommendations = []
inp = _inputs

if inp['bmi'] > 25:
    recommendations.append("**Overweight status**: Increases metabolic PCOS risk — consider weight management")
if inp['pulse'] > 90:
    recommendations.append("**Elevated resting heart rate**: May indicate metabolic stress or insulin resistance")
if inp['bp_sys'] > 130 or inp['bp_dia'] > 85:
    recommendations.append("**Elevated blood pressure**: Associated with metabolic syndrome and PCOS")
if inp['cycle'] < 21 or inp['cycle'] > 35:
    recommendations.append("**Irregular menstrual cycle**: Common PCOS indicator — track cycle patterns")

if _feature_set == 'Full Model':
    if inp['follicle_r'] and inp['follicle_l'] and (inp['follicle_r'] + inp['follicle_l']) > 20:
        recommendations.append("**High follicle count**: Polycystic ovary morphology — key PCOS diagnostic feature")
    if inp['amh'] and inp['amh'] > 7:
        recommendations.append("**Elevated AMH**: Increased ovarian reserve — common in PCOS")
    if inp['lh'] and inp['fsh'] and inp['lh'] / (inp['fsh'] + 0.1) > 2:
        recommendations.append("**Elevated LH/FSH ratio**: Hormonal imbalance typical of PCOS (>3 highly suggestive)")
else:
    symptom_count = sum(int(bool(inp[k])) for k in ['weight_gain', 'hair_growth', 'skin_darkening', 'hair_loss', 'pimples'])
    if inp['weight_gain']:
        recommendations.append("**Weight gain**: Suggests possible insulin resistance — core PCOS feature")
    if inp['hair_growth']:
        recommendations.append("**Hirsutism**: Sign of androgen excess typical of PCOS")
    if inp['skin_darkening']:
        recommendations.append("**Skin darkening**: May indicate insulin resistance (acanthosis nigricans)")
    if inp['hair_loss']:
        recommendations.append("**Hair loss**: Indicates elevated androgens — common PCOS symptom")
    if inp['pimples']:
        recommendations.append("**Acne/pimples**: Sign of hormonal imbalance")
    if symptom_count >= 3:
        recommendations.append("**Multiple PCOS symptoms**: Strong indicator — clinical evaluation recommended")
    if not inp['fast_food'] and inp['exercise']:
        recommendations.append("**Good lifestyle factors**: Regular exercise and healthy diet support PCOS management")

if risk_prob >= 0.50:
    if _feature_set == 'Full Model':
        recommendations.append("**High risk**: Recommend comprehensive PCOS evaluation by endocrinologist")
    else:
        recommendations.append("**High risk**: Non-invasive screening suggests PCOS — recommend clinical evaluation with blood tests & ultrasound")
elif risk_prob >= 0.20:
    if _feature_set == 'Full Model':
        recommendations.append("**Medium risk**: Monitor for PCOS symptoms and hormonal markers — consider repeat testing")
    else:
        recommendations.append("**Medium risk**: Additional testing recommended — consider bloodwork and pelvic ultrasound")
else:
    recommendations.append("**Low risk**: Current indicators suggest low PCOS probability — routine monitoring suggested")

for rec in recommendations:
    st.markdown(f"- {rec}")

if not recommendations:
    st.info("No specific recommendations based on current values.")
