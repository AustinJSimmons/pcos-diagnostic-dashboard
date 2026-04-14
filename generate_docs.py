from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Colour palette ──────────────────────────────────────────────────────────
PINK       = colors.HexColor("#C2185B")
PURPLE     = colors.HexColor("#7B1FA2")
LIGHT_PINK = colors.HexColor("#FCE4EC")
BLUSH      = colors.HexColor("#FDF4F8")
DARK       = colors.HexColor("#1C1C2E")
GREY       = colors.HexColor("#666666")
WHITE      = colors.white

W, H = A4

doc = SimpleDocTemplate(
    "PCOS_Dashboard_Changes.pdf",
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2.2*cm, bottomMargin=2*cm,
)

styles = getSampleStyleSheet()

# Custom styles
def S(name, **kw):
    return ParagraphStyle(name, **kw)

title_style = S("Title2", fontSize=26, textColor=PINK,
                fontName="Helvetica-Bold", spaceAfter=4, leading=30)
subtitle_style = S("Sub", fontSize=11, textColor=GREY,
                   fontName="Helvetica", spaceAfter=16)
section_style = S("Section", fontSize=14, textColor=PURPLE,
                  fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6)
subsection_style = S("Subsec", fontSize=11, textColor=PINK,
                     fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
body_style = S("Body2", fontSize=9.5, textColor=DARK,
               fontName="Helvetica", leading=15, spaceAfter=4)
bullet_style = S("Bullet", fontSize=9.5, textColor=DARK,
                 fontName="Helvetica", leading=15, leftIndent=14,
                 firstLineIndent=-10, spaceAfter=2)
code_style = S("Code2", fontSize=8.5, textColor=colors.HexColor("#3D1A5C"),
               fontName="Courier", leading=13, backColor=colors.HexColor("#F3E5F5"),
               leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=6)
tag_style = S("Tag", fontSize=8, textColor=WHITE,
              fontName="Helvetica-Bold", backColor=PINK,
              borderPadding=(2, 6, 2, 6))

def bullet(text):
    return Paragraph(f"• {text}", bullet_style)

def body(text):
    return Paragraph(text, body_style)

def section(text):
    return Paragraph(text, section_style)

def subsection(text):
    return Paragraph(text, subsection_style)

def divider():
    return HRFlowable(width="100%", thickness=1,
                      color=colors.HexColor("#EDD6F5"), spaceAfter=6, spaceBefore=6)

def tag_table(tags):
    cells = [[Paragraph(t, S(f"T{i}", fontSize=7.5, textColor=WHITE,
                              fontName="Helvetica-Bold")) for t, i in zip(tags, range(len(tags)))]]
    col_w = [(len(t)*5 + 16) for t in tags]
    tbl = Table(cells, colWidths=col_w, rowHeights=16)
    colors_list = [PINK, PURPLE, colors.HexColor("#AD1457"),
                   colors.HexColor("#6A1B9A"), PINK, PURPLE]
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (i, 0), (i, 0), colors_list[i % len(colors_list)])
        for i in range(len(tags))
    ] + [
        ('TEXTCOLOR', (0, 0), (-1, -1), WHITE),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return tbl


# ── Page background via canvas callback ─────────────────────────────────────
def draw_background(canvas, doc):
    canvas.saveState()
    # Soft blush background
    canvas.setFillColor(colors.HexColor("#FDF4F8"))
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Pink top bar
    canvas.setFillColor(PINK)
    canvas.rect(0, H - 1.1*cm, W, 1.1*cm, fill=1, stroke=0)
    # Footer bar
    canvas.setFillColor(colors.HexColor("#F8BBD9"))
    canvas.rect(0, 0, W, 0.8*cm, fill=1, stroke=0)
    # Footer text
    canvas.setFillColor(PINK)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(W/2, 0.28*cm,
        f"PCOS Diagnostic Dashboard — Development Log   |   Page {doc.page}")
    canvas.restoreState()


# ── Content ──────────────────────────────────────────────────────────────────
story = []

# Cover block
story.append(Spacer(1, 0.6*cm))
story.append(Paragraph("🏥 PCOS Diagnostic Dashboard", title_style))
story.append(Paragraph("Development Changes & Feature Documentation", subtitle_style))
story.append(Paragraph("March 2026  ·  Prepared by Sarthak Monga", subtitle_style))
story.append(divider())
story.append(Spacer(1, 0.3*cm))

# ── 1. Overview ──────────────────────────────────────────────────────────────
story.append(section("1. Project Overview"))
story.append(body(
    "This document summarises all changes and new features added to the PCOS Diagnostic "
    "Dashboard during a single development session. The dashboard is a multi-page Streamlit "
    "application built on a clinical dataset of 541 patients (177 PCOS, 364 controls) with "
    "41 features. It provides three analytical tools: Phenotype Explorer, Risk Calculator, "
    "and Feature Impact Analysis."
))
story.append(Spacer(1, 0.2*cm))

# ── 2. Dependency & Environment Fixes ───────────────────────────────────────
story.append(section("2. Environment & Dependency Fixes"))

story.append(subsection("2.1 Claude Code Upgrade"))
story.append(bullet("Upgraded Claude Code via Homebrew: <b>v2.1.84 → v2.1.85</b>"))

story.append(subsection("2.2 XGBoost / OpenMP Fix"))
story.append(body(
    "XGBoost failed to load on macOS because the OpenMP runtime (libomp) was missing. "
    "The library is installed as keg-only by Homebrew and was not symlinked."
))
story.append(bullet("Installed <b>libomp v22.1.2</b> via <font name='Courier'>brew install libomp</font>"))
story.append(bullet("Force-reinstalled xgboost via pip to relink against the new libomp"))

story.append(subsection("2.3 Streamlit Deprecation Warning"))
story.append(bullet(
    "Replaced all <font name='Courier'>st.pyplot(fig, use_container_width=True)</font> "
    "calls with <font name='Courier'>st.pyplot(fig, width='stretch')</font> across all pages"
))
story.append(Spacer(1, 0.2*cm))

# ── 3. New Features ──────────────────────────────────────────────────────────
story.append(section("3. New Features Added"))

story.append(subsection("3.1 XGBoost Model in Risk Calculator"))
story.append(body(
    "The Risk Calculator previously offered only Logistic Regression. XGBoost was added "
    "as a second algorithm option, giving four model combinations."
))
story.append(bullet("Model selection restructured into two independent radio buttons:"))
story.append(bullet("  — <b>Feature Set</b>: Full Model (9 features) or Non-Invasive Model (17 features)"))
story.append(bullet("  — <b>Algorithm</b>: Logistic Regression or XGBoost"))
story.append(bullet(
    "XGBoost trained with <font name='Courier'>scale_pos_weight</font> to handle "
    "the ~2:1 class imbalance (controls vs PCOS)"
))
story.append(bullet(
    "Feature contribution chart uses <font name='Courier'>feature_importances_</font> "
    "(gain-based) for XGBoost vs coefficients for Logistic Regression"
))

story.append(subsection("3.2 Patient-Specific Risk Contributions"))
story.append(body(
    "The 'Contributing Risk Factors' chart was changed from a static model-level feature "
    "importance display to a patient-specific contribution chart that updates with every run."
))
story.append(bullet(
    "<b>Logistic Regression</b>: contribution = <font name='Courier'>coef[i] × scaled_input[i]</font> "
    "— the signed contribution to log-odds for this patient"
))
story.append(bullet(
    "<b>XGBoost</b>: contribution = <font name='Courier'>importance[i] × scaled_deviation[i]</font> "
    "— how far this patient deviates from the population mean, weighted by feature importance"
))
story.append(bullet("Pink bars = increases PCOS risk; Purple bars = reduces PCOS risk"))

story.append(subsection("3.3 Calculate Button (Risk Calculator)"))
story.append(body(
    "Previously the risk score was computed on every widget interaction. A 'Calculate Risk Score' "
    "button was added so results only appear after explicit user action."
))
story.append(bullet("Results stored in <font name='Courier'>st.session_state</font> and persist between reruns"))
story.append(bullet("Informational message shown until the button is clicked for the first time"))

story.append(subsection("3.4 Model Comparison (Feature Impact Page)"))
story.append(body(
    "A new 'Model Comparison' analysis type was added to the Feature Impact page, comparing "
    "Logistic Regression vs XGBoost across both feature sets using 5-fold stratified cross-validation."
))
story.append(bullet("Metrics reported: Accuracy, ROC-AUC, F1 Score (mean ± SD across folds)"))
story.append(bullet("Summary table + grouped bar charts with error bars"))
story.append(bullet(
    "Four combinations: Full/Non-Invasive × LogReg/XGBoost — "
    "results cached with <font name='Courier'>@st.cache_data</font>"
))

story.append(subsection("3.5 Navigation Buttons on Home Page"))
story.append(body(
    "Clickable page links added to each card in the Navigation Guide section using "
    "<font name='Courier'>st.page_link()</font>, so users can navigate directly from the home page."
))
story.append(Spacer(1, 0.2*cm))

# ── 4. Home Page Rewrite ─────────────────────────────────────────────────────
story.append(section("4. Home Page Rewrite"))
story.append(body(
    "The About section was rewritten to be clinically meaningful rather than a generic list."
))
story.append(bullet("Added PCOS epidemiology context (8–13% prevalence, 70% undiagnosed)"))
story.append(bullet("Explained the heterogeneous presentation: metabolic vs hormonal subtypes"))
story.append(bullet("Described the motivation for the non-invasive model in low-resource settings"))
story.append(bullet("Added a Limitations & Disclaimer section for research/educational use"))
story.append(bullet("Added Best Model AUC metric (94.7%) to Quick Statistics"))
story.append(Spacer(1, 0.2*cm))

# ── 5. UI & Theme Changes ────────────────────────────────────────────────────
story.append(section("5. UI & Theme Changes"))

story.append(subsection("5.1 Global Theme (config.toml)"))
story.append(body("Switched from dark mode to a light pink/lavender theme:"))
data = [
    ["Setting", "Before", "After"],
    ["primaryColor", "#FF69B4", "#C2185B"],
    ["backgroundColor", "#0E1117 (dark)", "#FDF4F8 (blush)"],
    ["secondaryBackgroundColor", "#161B22", "#FFFFFF"],
    ["textColor", "#C9D1D9", "#1C1C2E"],
]
tbl = Table(data, colWidths=[5*cm, 5*cm, 5*cm])
tbl.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#FDF4F8"), WHITE]),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#E8D0F0")),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(Spacer(1, 0.15*cm))
story.append(tbl)
story.append(Spacer(1, 0.2*cm))

story.append(subsection("5.2 Shared Styles Module (app/styles.py)"))
story.append(body(
    "A centralised <font name='Courier'>styles.py</font> module was created so all pages share "
    "one CSS definition and matplotlib theme. Every page calls <font name='Courier'>apply_styles()</font> "
    "at startup."
))
story.append(bullet("Metric cards: white card, pink left border, shadow, hover effect"))
story.append(bullet("Sidebar: pink gradient (deep rose → hot pink → blush)"))
story.append(bullet("Buttons: pink-to-purple gradient pill with shadow"))
story.append(bullet("Form inputs (number inputs, selects, multiselect): rounded corners, lavender border"))
story.append(bullet("DataFrames: rounded corners, soft shadow"))
story.append(bullet("Sidebar radio: semi-transparent background with light text for readability"))

story.append(subsection("5.3 Chart Styling (style_fig helper)"))
story.append(body(
    "A <font name='Courier'>style_fig(fig, axes)</font> helper was added to <font name='Courier'>styles.py</font> "
    "and applied to every matplotlib chart across all pages."
))
story.append(bullet("Figure backgrounds set to <b>transparent</b> — charts blend into the page gradient"))
story.append(bullet("Axes background: soft pink (<font name='Courier'>#FDF0F8</font>)"))
story.append(bullet("Top and right spines removed; remaining spines styled in lavender"))
story.append(bullet("CSS card wrapper: white background, rounded corners, pink shadow on chart containers"))
story.append(bullet("Global matplotlib rcParams updated: grid colour, tick colour, title colour, font"))
story.append(bullet("All chart colours updated to pink/purple palette (#E91E8C, #C2185B, #7B1FA2)"))

story.append(subsection("5.4 Title Emoji Fix"))
story.append(body(
    "The CSS gradient on h1 elements was clipping emoji characters, rendering them as broken "
    "coloured rectangles. Fixed by:"
))
story.append(bullet("Removing the gradient clip from the global h1 CSS rule"))
story.append(bullet(
    "Rendering all page titles with inline HTML: emoji outside the "
    "<font name='Courier'>&lt;span&gt;</font> that carries the gradient, "
    "so emoji renders normally while the text gets the gradient"
))
story.append(bullet("Applied to: Home, Phenotype Explorer, Risk Calculator, Feature Impact"))
story.append(Spacer(1, 0.2*cm))

# ── 6. File Summary ──────────────────────────────────────────────────────────
story.append(section("6. Files Modified / Created"))
files = [
    [".streamlit/config.toml", "Theme colours switched from dark to light pink"],
    ["app/styles.py", "NEW — shared CSS + matplotlib theme + style_fig helper"],
    ["app/Home.py", "Rewritten about section, title fix, nav buttons"],
    ["app/pages/1_Phenotype_Explorer.py", "Title fix, style_fig on all charts, colour updates"],
    ["app/pages/2_Risk_Calculator.py", "XGBoost model, Calculate button, patient-specific contributions"],
    ["app/pages/3_Feature_Impact.py", "XGBoost model comparison tab, title fix, style_fig on all charts"],
]
ftbl = Table(files, colWidths=[6.5*cm, 9*cm])
ftbl.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F3E5F5")),
    ('FONTNAME', (0, 0), (0, -1), 'Courier'),
    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 0), (-1, -1), [WHITE, colors.HexColor("#FDF4F8")]),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#E8D0F0")),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(ftbl)
story.append(Spacer(1, 0.4*cm))
story.append(divider())
story.append(body(
    "<i>All changes were implemented iteratively with live preview via two concurrent "
    "Streamlit instances (ports 8501 and 8502) for before/after comparison.</i>"
))

# ── Build ────────────────────────────────────────────────────────────────────
doc.build(story, onFirstPage=draw_background, onLaterPages=draw_background)
print("PDF generated: PCOS_Dashboard_Changes.pdf")
