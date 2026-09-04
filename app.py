import json
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from src import CropConfig, analyze_field

st.set_page_config(page_title="KhetGap | Precision Agriculture", layout="wide")

# --- SVG ICONS ---
def get_icon(svg_path, color="#18181B", size=18):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;">{svg_path}</svg>'

ICON_LEAF = '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>'
ICON_FOLDER = '<path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>'
ICON_RULER = '<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/>'
ICON_SETTINGS = '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1Z"/>'
ICON_INFO = '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'
ICON_BAR_CHART = '<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>'
ICON_RESTART = '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>'
ICON_TABLE = '<path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/>'

def h_title(svg, title, level=3):
    return f"<h{level} style='font-weight: 600; font-size: 1.1rem; color: #18181B; display: flex; align-items: center;'>{get_icon(svg)}{title}</h{level}>"

# --- ULTRA PREMIUM MINIMALIST UI/UX CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --kg-text: #18181B;
        --kg-muted: #3F3F46;
        --kg-border: #52525B;
        --kg-rule: #A1A1AA;
        --kg-bg: #FFFFFF;
        --kg-sidebar: #F4F4F5;
        --text-color: #18181B;
        --background-color: #FFFFFF;
        --secondary-background-color: #F4F4F5;
        --primary-color: #18181B;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--kg-text);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent;}

    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--kg-bg);
        color: var(--kg-text);
    }

    .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
    [data-testid="stWidgetLabel"],
    [data-testid="stExpander"] summary {
        color: var(--kg-text) !important;
    }

    .stApp p,
    [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] p {
        color: var(--kg-text);
    }

    .hero-container {
        padding: 2rem 0 3rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--kg-rule);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--kg-text);
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: var(--kg-text);
        font-weight: 500;
        letter-spacing: -0.01em;
    }

    [data-testid="stMetric"] {
        background: #FAFAFA;
        border: 1px solid var(--kg-border);
        border-radius: 4px;
        padding: 1.5rem;
        transition: border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--kg-text);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--kg-text) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricLabel"] * {
        color: var(--kg-text) !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: var(--kg-text) !important;
        letter-spacing: -0.03em;
    }

    .stButton button,
    .stDownloadButton button,
    button[data-testid^="stBaseButton"] {
        font-weight: 600 !important;
        border-radius: 4px !important;
        padding: 0.75rem 1.5rem !important;
        width: 100%;
        font-size: 1rem !important;
        letter-spacing: -0.01em;
        transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    }

    /* Filled zinc buttons: primary AND secondary. White labels in default AND hover.
       Hover invert was the visibility bug — Streamlit lightens the fill while span/p
       rules kept the opposite text color. */
    .stButton button[data-testid="stBaseButton-primary"],
    .stButton button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: #18181B !important;
        background-image: none !important;
        color: #FFFFFF !important;
        border: 1px solid #18181B !important;
    }

    .stButton button[data-testid="stBaseButton-primary"]:hover,
    .stButton button[data-testid="stBaseButton-primary"]:focus-visible,
    .stButton button[data-testid="stBaseButton-secondary"]:hover,
    .stButton button[data-testid="stBaseButton-secondary"]:focus-visible,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:focus-visible,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:focus-visible {
        background-color: #3F3F46 !important;
        color: #FFFFFF !important;
        border-color: #3F3F46 !important;
    }

    .stButton button[data-testid="stBaseButton-primary"] *,
    .stButton button[data-testid="stBaseButton-secondary"] *,
    button[data-testid="stBaseButton-primary"] *,
    button[data-testid="stBaseButton-secondary"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    .stDownloadButton button,
    .stDownloadButton button[data-testid^="stBaseButton"] {
        background-color: #FFFFFF !important;
        background-image: none !important;
        color: #18181B !important;
        border: 2px solid #18181B !important;
    }

    .stDownloadButton button:hover,
    .stDownloadButton button:focus-visible {
        background-color: #E4E4E7 !important;
        color: #18181B !important;
        border: 2px solid #18181B !important;
    }

    .stDownloadButton button * {
        color: #18181B !important;
        fill: #18181B !important;
    }

    button[data-testid^="stBaseButton"]:disabled,
    button[data-testid^="stBaseButton"]:disabled:hover,
    .stDownloadButton button:disabled {
        background-color: #F4F4F5 !important;
        color: #3F3F46 !important;
        border-color: #52525B !important;
        cursor: not-allowed;
    }

    button[data-testid^="stBaseButton"]:disabled * {
        color: #3F3F46 !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--kg-sidebar);
        border-right: 1px solid var(--kg-rule);
        color: var(--kg-text);
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: var(--kg-text) !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--kg-text) !important;
    }

    [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] *,
    [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stDownloadButton button,
    [data-testid="stSidebar"] .stDownloadButton button * {
        color: #18181B !important;
        fill: #18181B !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        border-bottom: 1px solid var(--kg-rule);
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        color: var(--kg-muted) !important;
        border-bottom: 2px solid transparent;
        transition: color 0.2s;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--kg-text) !important;
        border-bottom-color: var(--kg-text) !important;
    }

    .stRadio label, .stSelectbox label, .stSlider label, .stNumberInput label, .stCheckbox label {
        color: var(--kg-text) !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stImageCaption"],
    [data-testid="stImageCaption"] *,
    [data-testid="stCaption"],
    [data-testid="stCaption"] *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    .stCaption,
    figcaption {
        color: #18181B !important;
        font-weight: 600 !important;
    }

    [data-testid="stDataFrame"] {
        color: var(--kg-text);
        border: 1px solid var(--kg-border);
    }

    .stAlert {
        color: var(--kg-text);
        border: 1px solid var(--kg-border);
    }

    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #52525B !important;
        border-radius: 4px;
        color: var(--kg-text) !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: var(--kg-text) !important;
        font-weight: 600 !important;
    }

    .kg-meta {
        color: #18181B !important;
        font-size: 0.85rem;
        font-weight: 600;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        color: #18181B !important;
        border-color: #52525B !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">{get_icon(ICON_LEAF, size=32)} KhetGap Precision Analytics</div>
    <div class="hero-subtitle">Automated Sugarcane Row Detection & Planting Gap Estimation</div>
</div>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
def reset_demo():
    for key in ['analysis_result', 'processed_image']:
        if key in st.session_state:
            del st.session_state[key]

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    # 1. Reset Button for live demos
    st.button("Reset Analysis", on_click=reset_demo, use_container_width=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #A1A1AA; margin: 1rem 0 2rem 0;'>", unsafe_allow_html=True)

    # 2. Upload and Sample Data
    st.markdown(h_title(ICON_FOLDER, "Data Source"), unsafe_allow_html=True)
    input_method = st.radio("Select Image Source", ("Sample Fixtures", "Upload Image"), label_visibility="collapsed")
    
    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an RGB aerial image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.markdown(f"<span class='kg-meta'>Detected Dimensions: {image.width}x{image.height} px</span>", unsafe_allow_html=True)
            except Exception:
                st.error("Invalid image format.")
                image = None
    else:
        sample_dir = Path("data/synthetic")
        sample_files = sorted([f.name for f in sample_dir.iterdir() if f.suffix.lower() in {".png", ".jpg"}]) if sample_dir.exists() else []
        if sample_files:
            selected_sample = st.selectbox("Select Sample", sample_files)
            try:
                image = Image.open(sample_dir / selected_sample).convert("RGB")
                st.markdown(f"<span class='kg-meta'>Detected Dimensions: {image.width}x{image.height} px</span>", unsafe_allow_html=True)
            except Exception:
                st.error("Error loading sample.")
                image = None
        else:
            st.warning("No sample files found in data/synthetic/")

    st.markdown("<hr style='border: none; border-top: 1px solid #A1A1AA; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # 3. Calibration
    st.markdown(h_title(ICON_RULER, "Calibration & Crop"), unsafe_allow_html=True)
    
    # Crop Profile selection to dynamically set Advanced parameters
    crop_profile = st.selectbox("Crop Profile", ["Sugarcane", "Maize"])
    def_exg = 25
    def_min_gap = 18 if crop_profile == "Sugarcane" else 14
    def_occupancy = 0.18 if crop_profile == "Sugarcane" else 0.15
    def_row_spacing = 24 if crop_profile == "Sugarcane" else 15
    
    meters_per_pixel = st.number_input(
        "Pixel Scale (m/px)",
        value=0.0,
        step=0.01,
        format="%.3f",
        help="Set 0 if the image is uncalibrated. Gap detection then uses Min Gap Length in pixels.",
    )
    if meters_per_pixel <= 0:
        meters_per_pixel = None
        
    st.markdown("<hr style='border: none; border-top: 1px solid #A1A1AA; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # 4. Advanced Tuning
    st.markdown(h_title(ICON_SETTINGS, "Advanced Tuning"), unsafe_allow_html=True)
    
    with st.expander("Pipeline Parameters", expanded=False):
        exg_threshold = st.slider("ExG Threshold", 0, 100, def_exg, help="Greenness threshold for vegetation segmentation.")
        min_gap_px = st.slider("Min Gap Length (px)", 6, 100, def_min_gap, help="Minimum pixel length to be considered a planting gap. This bound is always applied, even when a metre scale is set.")
        occupancy_threshold = st.slider("Occupancy Threshold", 0.0, 1.0, def_occupancy, step=0.01, help="Baseline vegetation fraction used when deciding whether a row column is planted.")
        min_row_spacing_px = st.slider("Min Row Spacing (px)", 10, 100, def_row_spacing, help="Minimum pixel distance between adjacent rows.")

    with st.expander("Orientation Overrides", expanded=False):
        use_manual_angle = st.checkbox("Enable Manual Row Angle")
        manual_angle = st.slider("Row Angle (deg)", -90.0, 90.0, 0.0) if use_manual_angle else None

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)


# --- ANALYSIS EXECUTION ---
if run_button and image is not None:
    with st.spinner("Analyzing field data..."):
        img_array = np.asarray(image)
        config = CropConfig(
            exg_threshold=exg_threshold,
            min_gap_px=min_gap_px,
            occupancy_threshold=occupancy_threshold,
            min_row_spacing_px=min_row_spacing_px
        )
        overrides = {"row_angle_deg": manual_angle if use_manual_angle else None}
        
        result = analyze_field(
            img_array, 
            config, 
            meters_per_pixel=meters_per_pixel, 
            overrides=overrides
        )
        
        # Store results in session state to survive widget interactions
        st.session_state['analysis_result'] = result
        st.session_state['processed_image'] = image

# --- MAIN CONTENT AREA ---
if 'analysis_result' in st.session_state and 'processed_image' in st.session_state:
    result = st.session_state['analysis_result']
    processed_image = st.session_state['processed_image']
    
    if result.ok:
        # Top Metrics
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Rows Detected", result.metrics.get("rows_detected", 0))
        col2.metric("Gaps Detected", result.metrics.get("gaps_detected", 0))
        
        missing_m = result.metrics.get("missing_length_m")
        col3.metric("Missing Length", f"{missing_m:.2f} m" if missing_m is not None else "N/A")
        
        orientation = result.metrics.get("orientation_method", "N/A").title()
        col4.metric("Orientation", orientation)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if result.pattern_hints:
            for hint in result.pattern_hints:
                st.markdown(f"<div style='border-left: 2px solid #18181B; padding-left: 1rem; color: #3F3F46; margin-bottom: 1rem;'>{get_icon(ICON_INFO, size=14, color='#3F3F46')} Analysis Hint: {hint.get('description', '')}</div>", unsafe_allow_html=True)
        
        # Visual Tabs
        st.markdown(f"<br>{h_title(ICON_BAR_CHART, 'Visualization & Diagnostics')}", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Result Overlay", "Vegetation Mask", "Rotated Mask", "Row Bands"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(processed_image, caption="Input Image", use_container_width=True)
            with col_b:
                st.image(result.overlay_image, caption="AI Analysis Overlay", use_container_width=True)
        
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            vegetation = result.debug_images.get("occupancy_mask")
            if vegetation is None:
                vegetation = result.debug_images.get("mask")
            if vegetation is not None:
                st.image(
                    vegetation,
                    caption="Vegetation Mask (gaps from holes in this mask)",
                    use_container_width=True,
                )
            else:
                st.warning("Mask debug image not available.")
                
        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            if "rotated_mask" in result.debug_images:
                st.image(result.debug_images["rotated_mask"], caption="Orientation Corrected Mask", use_container_width=True)
            else:
                st.warning("Rotated Mask debug image not available.")
                
        with tab4:
            st.markdown("<br>", unsafe_allow_html=True)
            if "row_bands" in result.debug_images:
                st.image(result.debug_images["row_bands"], caption="Detected Row Bands", use_container_width=True)
            else:
                st.warning("Row Bands debug image not available.")


        # Gap Table
        st.markdown("<hr style='border: none; border-top: 1px solid #A1A1AA; margin: 3rem 0;'>", unsafe_allow_html=True)
        st.markdown(h_title(ICON_TABLE, 'Detected Gaps (Priority Sorted)'), unsafe_allow_html=True)
        
        df_gaps = pd.DataFrame()
        if result.gaps:
            gap_data = []
            for g in result.gaps:
                gap_data.append({
                    "Row ID": g.row_id,
                    "Gap Length (px)": g.length_px,
                    "Gap Length (m)": round(g.length_m, 2) if g.length_m is not None else "N/A"
                })
            df_gaps = pd.DataFrame(gap_data)
            # Sort by longest gap first
            df_gaps = df_gaps.sort_values(by="Gap Length (px)", ascending=False).reset_index(drop=True)
            st.dataframe(df_gaps, use_container_width=True)
        else:
            st.info("No gaps detected matching the minimum criteria.")


        st.markdown("<hr style='border: none; border-top: 1px solid #A1A1AA; margin: 3rem 0;'>", unsafe_allow_html=True)
        st.markdown(h_title(ICON_FOLDER, 'Export Reports'), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        # 1. Overlay Image Download
        from io import BytesIO
        img_buffer = BytesIO()
        Image.fromarray(result.overlay_image).save(img_buffer, format="PNG")
        col_export1.download_button(
            label="Download HD Overlay",
            data=img_buffer.getvalue(),
            file_name="khetgap_overlay.png",
            mime="image/png"
        )
        
        # 2. Text Summary Report
        report_lines = [
            "========================================",
            "      KHETGAP FIELD ANALYSIS REPORT      ",
            "========================================",
            f"Rows Detected:   {result.metrics.get('rows_detected', 0)}",
            f"Gaps Detected:   {result.metrics.get('gaps_detected', 0)}",
            f"Missing Length:  {missing_m:.2f} m" if missing_m else "Missing Length:  N/A (Uncalibrated)",
            f"Orientation:     {orientation}",
            "----------------------------------------",
            "PATTERN HINTS:"
        ]
        if result.pattern_hints:
            for hint in result.pattern_hints:
                report_lines.append(f"- {hint.get('description', '')}")
        else:
            report_lines.append("- No spatial patterns detected.")
            
        report_text = "\\n".join(report_lines)
        col_export2.download_button(
            label="Download Text Report",
            data=report_text,
            file_name="khetgap_summary.txt",
            mime="text/plain"
        )
        
        # 3. CSV Gap Download
        if not df_gaps.empty:
            csv_data = df_gaps.to_csv(index=False)
            col_export3.download_button(
                label="Download Gap CSV",
                data=csv_data,
                file_name="khetgap_table.csv",
                mime="text/csv"
            )
        
    else:
        st.error(f"Analysis failed: {result.status}")
        for err in result.errors:
            st.error(f"{err.get('code', 'Error')}: {err.get('message', '')}")
else:
    # Empty State
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; color: #18181B; font-weight: 600;'>{get_icon(ICON_FOLDER, size=24, color='#18181B')}<br><br>Please select or upload a field image in the sidebar and click <b>Run Analysis</b> to begin.</div>", unsafe_allow_html=True)
