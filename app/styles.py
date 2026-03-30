import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def apply_mpl_theme():
    mpl.rcParams.update({
        'figure.facecolor': 'none',       # transparent — blends with page bg
        'axes.facecolor': '#FDF0F8',       # very soft pink axes area
        'axes.edgecolor': '#F494C6',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelcolor': '#950F54',
        'axes.titlecolor': '#950F54',
        'axes.titleweight': 'bold',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.titlepad': 14,
        'xtick.color': '#666',
        'ytick.color': '#666',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'grid.color': '#F494C6',
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#F494C6',
        'legend.fontsize': 10,
        'figure.autolayout': False,
        'font.family': 'DejaVu Sans',
    })


def style_fig(fig, axes=None):
    """Apply consistent polish to a figure before rendering."""
    fig.patch.set_facecolor('none')   # transparent figure bg
    fig.patch.set_alpha(0.0)

    if axes is not None:
        ax_list = axes if hasattr(axes, '__iter__') else [axes]
        for ax in ax_list:
            if not ax.get_visible():
                continue
            ax.set_facecolor('#FDF0F8')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('#E8D0F0')
                ax.spines[spine].set_linewidth(1.2)

    fig.tight_layout(pad=2.0)


def apply_styles():
    apply_mpl_theme()
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #FDF4F8;
    }

    /* ── Hide default Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Page titles handled inline per page to avoid emoji clipping ── */
    h1 {
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.2rem !important;
        color: #950F54 !important;
    }

    /* ── Section headers ── */
    h2, h3 {
        color: #950F54 !important;
        font-weight: 600 !important;
    }

    /* ── Chart containers — card style with rounded corners ── */
    [data-testid="stImage"] {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 12px;
        box-shadow: 0 4px 24px rgba(234, 40, 141, 0.09);
        overflow: hidden;
    }
    [data-testid="stImage"] img {
        border-radius: 12px;
        display: block;
        width: 100%;
    }

    /* Also catch pyplot output wrapper */
    .element-container:has(img) > div {
        border-radius: 18px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 12px rgba(234, 40, 141, 0.08);
        border-left: 4px solid #EA288D;
        transition: box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 20px rgba(234, 40, 141, 0.15);
    }
    [data-testid="stMetricLabel"] {
        color: #888 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #1C1C2E !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #F494C6 !important;
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #950F54 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #950F54 !important;
        font-weight: 500;
    }
    [data-testid="stSidebarNav"] a {
        color: #950F54 !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 6px 12px;
        margin: 2px 0;
        transition: background 0.2s;
    }
    [data-testid="stSidebarNav"] a:hover {
        background: rgba(255,255,255,0.15) !important;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(255,255,255,0.2) !important;
        font-weight: 700 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #EA288D;
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        transition: opacity 0.2s, transform 0.1s;
        box-shadow: 0 4px 14px rgba(234, 40, 141, 0.3);
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* ── Radio buttons (main content) ── */
    .stRadio > div {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }

    /* ── Sidebar radio — transparent bg, light readable text ── */
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255, 255, 255, 0.12) !important;
        box-shadow: none !important;
        border: 1px solid rgba(255,255,255,0.2);
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label span {
        color: #950F54 !important;
        font-weight: 500 !important;
    }

    /* ── Select boxes ── */
    .stSelectbox > div > div {
        background: #FFFFFF;
        border-radius: 10px;
        border: 1.5px solid #E8D0F0;
    }

    /* ── Number inputs ── */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid #E8D0F0;
        background: #FFFFFF;
    }

    /* ── Multiselect ── */
    .stMultiSelect > div > div {
        background: #FFFFFF;
        border-radius: 10px;
        border: 1.5px solid #E8D0F0;
    }

    /* ── Checkboxes ── */
    .stCheckbox label span {
        font-weight: 500;
    }

    /* ── Info / warning / success / error boxes ── */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
        font-size: 0.9rem;
    }

    /* ── Divider ── */
    hr {
        border-color: #EDD6F5 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── DataFrames / tables ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    /* ── Content area ── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Custom card/text helpers ── */
    .pcos-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 4px 20px rgba(194, 24, 91, 0.08);
        margin-bottom: 16px;
    }
    .gradient-text {
        color: #EA288D;
        font-weight: 700;
    }
    .stat-pill {
        display: inline-block;
        background: #F494C6;
        color: #950F54;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    </style>
    """, unsafe_allow_html=True)
