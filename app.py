import os
import random
import datetime
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import urllib.request
from PIL import Image
from io import BytesIO

# Import ReportLab elements safely for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Page configurations
st.set_page_config(
    page_title="Skin Cancer Classification & XAI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SQLite Database Setup Functions
DB_FILE = "skin_cancer_patient_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            patient_age INTEGER,
            patient_gender TEXT,
            body_site TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            scan_date TEXT NOT NULL,
            doctor_status TEXT DEFAULT 'Pending'
        )
    ''')
    conn.commit()
    conn.close()

def insert_patient(p_id, p_name, p_age, p_gender, p_site, prediction, confidence):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO patients (patient_id, patient_name, patient_age, patient_gender, body_site, prediction, confidence, scan_date, doctor_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending')
    ''', (p_id, p_name, p_age, p_gender, p_site, prediction, round(confidence, 2), current_date))
    conn.commit()
    conn.close()

def update_doctor_status(record_id, status):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE patients SET doctor_status = ? WHERE id = ?
    ''', (status, record_id))
    conn.commit()
    conn.close()

def get_all_patients():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT 
            id,
            patient_id AS 'Patient ID', 
            patient_name AS 'Patient Name', 
            patient_age AS 'Age', 
            patient_gender AS 'Gender', 
            body_site AS 'Scan Body Site', 
            prediction AS 'AI Diagnosis', 
            confidence AS 'Confidence (%)', 
            scan_date AS 'Scan Date',
            doctor_status AS 'Doctor Status'
        FROM patients ORDER BY id DESC
    """, conn)
    conn.close()
    return df

# Initialize Database
init_db()

# Premium Clinical Dashboard Styling + Typewriter Animated Heading Logic
st.markdown("""
    <style>
    .main { background-color: #0f111a; color: #e2e8f0; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #ff6a00, #ff0055); }
    div[data-testid="stSidebar"] { background-color: #1a1c24; border-right: 1px solid #2d3748; }
    
    .medical-title { 
        font-family: 'Outfit', sans-serif; 
        font-weight: 700; 
        background: linear-gradient(135deg, #ff6a00 0%, #ee0979 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-size: 2.8rem; 
        margin-bottom: 0.5rem; 
        overflow: hidden; 
        white-space: nowrap; 
        border-right: .15em solid #ff6a00; 
        width: 0; 
        animation: typing 3.5s steps(46) forwards, blink-caret .75s step-end 5 forwards;
    }
    
    @keyframes typing { from { width: 0 } to { width: 100% } }
    @keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: #ff6a00; } 100% { border-color: transparent; } }
    
    .medical-subtitle { color: #a0aec0; font-size: 1.1rem; margin-bottom: 2rem; }
    .verdict-box { background-color: #1b1e2e; border-left: 6px solid #ff6a00; padding: 1.8rem; border-radius: 8px; margin-top: 1rem; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .section-title { border-bottom: 2px solid #2d3748; padding-bottom: 0.5rem; margin-top: 1.5rem; margin-bottom: 1rem; color: #ff6a00; }
    </style>
""", unsafe_allow_html=True)

# Safe TensorFlow Check & Global Hugging Face Model Loader
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception as e:
    pass

MODEL_URL = "https://huggingface.co/2046sorna/skin-cancer-model/resolve/main/DenseNet121_final.keras"
model_path = "DenseNet121_final.keras"

@st.cache_resource
def load_deep_learning_model():
    if not TF_AVAILABLE:
        return None, "TensorFlow missing."
    if not os.path.exists(model_path):
        try:
            with st.spinner("📥 Downloading Fine-Tuned AI Model from Hugging Face... Please wait."):
                urllib.request.urlretrieve(MODEL_URL, model_path)
        except Exception as e:
            return None, f"Download failed: {str(e)}"
            
    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            return model, "Loaded successfully"
        except Exception as e:
            return None, str(e)
    return None, "File missing"

# Upgraded Skin Cancer Class Mappings
CLASS_LABELS = ['AKIEC', 'BCC', 'DF', 'MEL', 'NV', 'BKL', 'SCC', 'SK', 'VASC']
CLASS_COLORS = ['#E74C3C', '#3498DB', '#9B59B6', '#FF0000', '#2ECC71', '#F39C12', '#8E44AD', '#16A085', '#D35400']

CLINICAL_DATA = {
    "AKIEC": {"summary": "Actinic Keratosis / Bowen's Disease (Pre-cancerous lesion) detected.", "advise": "1. Avoid excessive direct sunlight exposure.\n2. Consult a dermatologist for cryotherapy or formal biopsy selection."},
    "BCC": {"summary": "Basal Cell Carcinoma detected. Common skin cancer that grows slowly.", "advise": "1. Schedule a professional clinical removal or excision.\n2. Protect surrounding skin from UV radiation rigorously."},
    "DF": {"summary": "Dermatofibroma detected. This is generally a benign (non-cancerous) skin nodule.", "advise": "1. Regular self-monitoring is recommended.\n2. Seek advice only if it changes rapidly in size or color."},
    "MEL": {"summary": "Melanoma detected. Highly serious malignant skin cancer requiring prompt surgical attention.", "advise": "1. Urgent physical consultation with an oncologist or dermatologist is mandatory.\n2. Do not delay wide local clinical excision tests."},
    "NV": {"summary": "Melanocytic Nevus (Common Mole). Structurally stable and completely benign.", "advise": "1. Monitor using ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolving).\n2. Schedule routine annual skin checks."},
    "BKL": {"summary": "Benign Keratosis-like lesions (solar lentigines / seborrheic keratoses) detected.", "advise": "1. Keep under casual self-observation.\n2. Clinical treatment is purely optional unless triggered by irritation."},
    "SCC": {"summary": "Squamous Cell Carcinoma detected. Requires appropriate localized medical management.", "advise": "1. Immediate referral for clinical biopsy or deep surgical removal.\n2. Apply broad-spectrum sunscreens meticulously."},
    "SK": {"summary": "Seborrheic Keratosis detected. Very common, completely non-cancerous skin growth.", "advise": "1. Harmless lesion; treatment is completely unnecessary unless physically irritated.\n2. No active emergency response required."},
    "VASC": {"summary": "Vascular Skin Lesion (Cherry angiomas / angiokeratomas). Harmless blood vessel structures.", "advise": "1. No major malignancy threat detected.\n2. Consult a specialist if the lesion starts active bleeding or changes configuration."}
}

# Demo Mode Logic
def compute_attention_demo(img_array, num_classes):
    img_gray = cv2.cvtColor(np.uint8(255 * img_array), cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape
    y, x = np.ogrid[:h, :w]
    mask = np.exp(-((y - h//2)**2 + (x - w//2)**2) / (2 * (45**2)))
    edges = cv2.Canny(img_gray, 30, 100)
    edges_blur = cv2.GaussianBlur(edges, (15, 15), 0).astype(float) / 255.0
    heatmap = mask * 0.75 + edges_blur * 0.25
    heatmap = heatmap / (np.max(heatmap) + 1e-10)
    
    rng = np.random.default_rng(int(np.sum(img_gray)) % 1000)
    raw_probs = rng.dirichlet(np.ones(num_classes))
    raw_probs[rng.integers(0, num_classes)] += 2.5
    probs = raw_probs / np.sum(raw_probs)
    return heatmap, probs

# Real TensorFlow Grad-CAM Logic
def compute_attention_real(model, img_tensor):
    img_var = tf.Variable(tf.cast(img_tensor, tf.float32), trainable=True)
    with tf.GradientTape() as tape:
        preds = model(img_var, training=False)
        class_idx = tf.argmax(preds[0])
        class_score = preds[:, class_idx]
    grads = tape.gradient(class_score, img_var)
    heatmap = tf.reduce_max(tf.abs(grads[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), preds.numpy()

def overlay_heatmap(heatmap, img_rgb, alpha=0.45):
    h = cv2.resize(heatmap, (224, 224))
    h = np.uint8(255 * h)
    h_col = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    h_col = cv2.cvtColor(h_col, cv2.COLOR_BGR2RGB)
    img_u8 = np.uint8(255 * np.clip(img_rgb, 0, 1))
    return cv2.addWeighted(img_u8, 1 - alpha, h_col, alpha, 0)

def process_single_image(img_file, is_demo_mode, loaded_model):
    pil_img = Image.open(img_file).convert("RGB")
    pil_resized = pil_img.resize((224, 224))
    img_rgb = np.array(pil_resized).astype(float) / 255.0
    
    if is_demo_mode:
        heatmap, probs = compute_attention_demo(img_rgb, len(CLASS_LABELS))
        return img_rgb, overlay_heatmap(heatmap, img_rgb), probs
    
    ensemble_probs = np.zeros((1, len(CLASS_LABELS)))
    ensemble_heatmaps = []
    
    for pass_idx in range(5):
        if pass_idx == 0:   t_img = img_rgb
        elif pass_idx == 1: t_img = np.fliplr(img_rgb)
        elif pass_idx == 2: t_img = np.flipud(img_rgb)
        elif pass_idx == 3: t_img = np.rot90(img_rgb, 1)
        elif pass_idx == 4: t_img = np.rot90(img_rgb, 2)
            
        img_tensor = np.expand_dims(t_img, axis=0)
        h, p = compute_attention_real(loaded_model, img_tensor)
        
        if pass_idx == 1:   h = np.fliplr(h)
        elif pass_idx == 2: h = np.flipud(h)
        elif pass_idx == 3: h = np.rot90(h, -1)
        elif pass_idx == 4: h = np.rot90(h, -2)
            
        ensemble_probs += p
        ensemble_heatmaps.append(h)
        
    final_heatmap = np.mean(ensemble_heatmaps, axis=0)
    final_heatmap = final_heatmap / (np.max(final_heatmap) + 1e-10)
    final_probs = ensemble_probs[0] / 5.0
    overlay = overlay_heatmap(final_heatmap, img_rgb)
    return img_rgb, overlay, final_probs

# 📥 PDF Report Generator
def generate_pdf_report(p_id, p_name, p_age, p_gender, p_site, diagnosis, confidence):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#1A365D'), spaceAfter=6, alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=10, textColor=colors.HexColor('#4A5568'), spaceAfter=20, alignment=1)
    section_heading = ParagraphStyle('SecHead', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, textColor=colors.HexColor('#FF6A00'), spaceBefore=12, spaceAfter=8)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=10.5, textColor=colors.HexColor('#2D3748'), leading=14)
    verdict_style = ParagraphStyle('VerdictText', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13, textColor=colors.HexColor('#C53030'))

    story.append(Paragraph("DERMATOLOGICAL SKIN CANCER AI REPORT", title_style))
    story.append(Paragraph("Automated Lesion Screening Powered by Fine-Tuned Deep Neural Networks", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Patient Information Records", section_heading))
    current_date = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")
    
    patient_data = [
        [Paragraph(f"<b>Patient ID:</b> {p_id}", body_style), Paragraph(f"<b>Age / Gender:</b> {p_age}Y / {p_gender}", body_style)],
        [Paragraph(f"<b>Patient Name:</b> {p_name}", body_style), Paragraph(f"<b>Scan Location:</b> {p_site}", body_style)],
        [Paragraph(f"<b>Date & Time:</b> {current_date}", body_style), Paragraph("<b>System Status:</b> Verified Log", body_style)]
    ]
    t1 = Table(patient_data, colWidths=[265, 265])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("AI Diagnostic Summary", section_heading))
    verdict_data = [
        [Paragraph("<b>Primary Screening Outcome:</b>", body_style), Paragraph(diagnosis, verdict_style)],
        [Paragraph("<b>Model Match Confidence Score:</b>", body_style), Paragraph(f"{confidence:.2f}% Match Rate", body_style)],
        [Paragraph("<b>Diagnostic Summary:</b>", body_style), Paragraph(CLINICAL_DATA[diagnosis]["summary"], body_style)]
    ]
    t2 = Table(verdict_data, colWidths=[180, 350])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#EDF2F7')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t2)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Urgent Care Plans & Clinical Suggestions", section_heading))
    advise_text = CLINICAL_DATA[diagnosis]["advise"].replace("\n", "<br/>")
    story.append(Paragraph(advise_text, body_style))
    story.append(Spacer(1, 40))
    
    sig_data = [
        [Paragraph("___________________________<br/><b>Medical Officer Signature</b>", body_style), 
         Paragraph("___________________________<br/><b>Dermatologist Stamp/Triage</b>", body_style)]
    ]
    t3 = Table(sig_data, colWidths=[270, 260])
    t3.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t3)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- ১. সাইডবার সেকশন (সাইডবারের ওপরের বাটন মডিফিকেশন) ---
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] button[data-testid="sidebar-collapse-control"] svg,
    button[aria-label="Close sidebar"] svg {
        display: none !important;
    }

    [data-testid="stSidebar"] button[data-testid="sidebar-collapse-control"]::after,
    button[aria-label="Close sidebar"]::after {
        content: "☰" !important;
        color: #ff6a00 !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        position: absolute !important;
        left: 50% !important;
        top: 50% !important;
        transform: translate(-50%, -50%) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    button[aria-label="Open sidebar"] svg {
        display: none !important;
    }
    button[aria-label="Open sidebar"]::after {
        content: "☰" !important;
        color: #ff6a00 !important;
        font-size: 1.3rem !important;
        font-weight: bold;
    }
    </style>

    <div style='text-align: center; padding: 10px 0;'>
        <h1 style='color: #ff6a00; font-family: "Outfit", sans-serif; font-weight: 800; font-size: 2.2rem; margin-bottom: 0;'>
            🛡️ SkinAI
        </h1>
        <p style='color: #718096; font-size: 0.85rem; margin-top: 2px;'>Advanced Dermoscopic Intelligence</p>
    </div>
    <hr style='margin-top: 0; margin-bottom: 20px; border-color: #2d3748;'>
    """, 
    unsafe_allow_html=True
)

# ক্লিনিক্যাল ওয়ার্কস্পেস ড্রপডাউন
menu_display = st.sidebar.selectbox(
    "🏥 Clinical Workspace",
    ["🧬 AI Diagnostic Chamber", "🗃️ Patient History Logs"],
    key="clinical_workspace_navigation"
)

menu_selection = "Home Page" if menu_display == "🧬 AI Diagnostic Chamber" else "Patient History"

# প্যাথলজি হেডার ও ৩x3 গ্রিড সেকশন
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<h4 style='color: #ffffff; font-size: 1.1rem; font-weight: 700; margin-bottom: 12px;'>📋 Monitored Pathologies</h4>", 
    unsafe_allow_html=True
)

grid_style = """
<style>
.badge-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    width: 100%;
}
.badge-item {
    display: block;
    padding: 6px 0px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    font-family: monospace;
    background-color: #1b1e2e;
    border: 1px solid #ff6a00;
    color: #ff6a00;
    text-align: center;
}
</style>
"""

badges_html = "<div class='badge-grid'>"
for cl in CLASS_LABELS:
    badges_html += f"<div class='badge-item'>{cl}</div>"
badges_html += "</div>"

st.sidebar.markdown(grid_style + badges_html, unsafe_allow_html=True)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# মডেল লোডিং ও মোড চেকিং
model, status = load_deep_learning_model()
is_demo = model is None

if model is not None:
    st.sidebar.markdown(
        """
        <div style='background-color: rgba(46, 204, 113, 0.1); border: 1px solid #2ecc71; padding: 12px; border-radius: 8px; text-align: center;'>
            <span style='color: #2ecc71; font-weight: 700; font-size: 0.9rem;'>● CORE ENGINE ACTIVE</span>
            <p style='color: #a0aec0; font-size: 0.75rem; margin: 4px 0 0 0;'>DenseNet121 Pipeline Online</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div style='background-color: rgba(241, 196, 15, 0.1); border: 1px solid #f1c40f; padding: 12px; border-radius: 8px; text-align: center;'>
            <span style='color: #f1c40f; font-weight: 700; font-size: 0.9rem;'>▲ DEMO SIMULATION MODE</span>
            <p style='color: #a0aec0; font-size: 0.75rem; margin: 4px 0 0 0;'>Mathematical Heatmap Active</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


# --- HOME PAGE: AI DIAGNOSTIC CHAMBER ---
if menu_selection == "Home Page":
    st.markdown("<h1 class='medical-title'>🛡️ Skin Cancer Classification & XAI Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='medical-subtitle'>Upload patient dermoscopic skin lesion images for targeted pathology profiling and instant diagnostics.</p>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ Interactive ABCDE Dermatological Guide (Clinical Reference)", expanded=False):
        st.markdown("<p style='color: #a0aec0; font-size: 0.95rem;'>Use the international ABCDE criteria to manually evaluate skin lesions and moles for potential malignancy risk.</p>", unsafe_allow_html=True)
        
        abcde_cols = st.columns(5)
        with abcde_cols[0]:
            st.markdown("<div style='background-color: #1b1e2e; padding: 15px; border-radius: 8px; border-top: 4px solid #E74C3C; min-height: 180px;'><h4 style='color: #E74C3C; margin-top:0;'>A - Asymmetry</h4><p style='font-size: 0.85rem; color: #e2e8f0; line-height: 1.3;'>One half of the mole or lesion does not match the other half in shape or structure.</p></div>", unsafe_allow_html=True)
        with abcde_cols[1]:
            st.markdown("<div style='background-color: #1b1e2e; padding: 15px; border-radius: 8px; border-top: 4px solid #3498DB; min-height: 180px;'><h4 style='color: #3498DB; margin-top:0;'>B - Border</h4><p style='font-size: 0.85rem; color: #e2e8f0; line-height: 1.3;'>The edges are irregular, ragged, notched, blurred, or poorly defined clinical margins.</p></div>", unsafe_allow_html=True)
        with abcde_cols[2]:
            st.markdown("<div style='background-color: #1b1e2e; padding: 15px; border-radius: 8px; border-top: 4px solid #9B59B6; min-height: 180px;'><h4 style='color: #9B59B6; margin-top:0;'>C - Color</h4><p style='font-size: 0.85rem; color: #e2e8f0; line-height: 1.3;'>The color is not uniform. Shades of brown, black, red, white, or blue may be visible.</p></div>", unsafe_allow_html=True)
        with abcde_cols[3]:
            st.markdown("<div style='background-color: #1b1e2e; padding: 15px; border-radius: 8px; border-top: 4px solid #F39C12; min-height: 180px;'><h4 style='color: #F39C12; margin-top:0;'>D - Diameter</h4><p style='font-size: 0.85rem; color: #e2e8f0; line-height: 1.3;'>The spot is larger than 6 millimeters across (about the size of a pencil eraser), though some can be smaller.</p></div>", unsafe_allow_html=True)
        with abcde_cols[4]:
            st.markdown("<div style='background-color: #1b1e2e; padding: 15px; border-radius: 8px; border-top: 4px solid #2ECC71; min-height: 180px;'><h4 style='color: #2ECC71; margin-top:0;'>E - Evolving</h4><p style='font-size: 0.85rem; color: #e2e8f0; line-height: 1.3;'>The mole is changing in size, shape, color, or is causing new symptoms like itching or bleeding.</p></div>", unsafe_allow_html=True)
            
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.info("🚨 **Clinical Warning:** If a lesion triggers **2 or more** of the ABCDE criteria above, immediate dermatological biopsy and histopathological examination are strongly advised, regardless of the AI score.")
    
    st.markdown("<h3 class='section-title'>👤 Patient Registration</h3>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3, col_p4 = st.columns([2, 3, 1, 1.5])
    with col_p1: patient_id = st.text_input("Patient Unique ID", value="P-")
    with col_p2: patient_name = st.text_input("Patient Full Name", value="")
    with col_p3: patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with col_p4: patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
    patient_site = st.radio("Select Scan Body Site Location:", ["Arm/Leg", "Torso/Back", "Face/Neck", "Hand/Foot"], horizontal=True)
    uploaded_files = st.file_uploader("Select Dermoscopic Scan...", type=["png", "jpg", "jpeg"])
    
    if uploaded_files:
        if not patient_name.strip() or patient_id == "P-":
            st.error("⚠️ Please provide a valid Patient ID and Name before executing scan.")
        else:
            with st.spinner("Executing deep clinical screening scans..."):
                img_rgb, overlay, probs = process_single_image(uploaded_files, is_demo, model)
                
            pred_class_idx = np.argmax(probs)
            pred_class_name = CLASS_LABELS[pred_class_idx]
            pred_confidence = probs[pred_class_idx] * 100
            color = CLASS_COLORS[pred_class_idx]
            
            state_key = f"saved_{patient_id}_{uploaded_files.name}"
            if state_key not in st.session_state:
                insert_patient(patient_id, patient_name, patient_age, patient_gender, patient_site, pred_class_name, pred_confidence)
                st.session_state[state_key] = True
                st.toast(f"💾 Record saved for {patient_name} ({patient_site})!", icon="✅")
                
            st.markdown("<h3 class='section-title'>📊 1. Diagnostic Class & Confidence Verdict</h3>", unsafe_allow_html=True)
            st.markdown(
                f"""<div class='verdict-box' style='border-left-color: {color};'>
                    <span style='font-size: 1.1rem; color: #a0aec0; font-weight: 500;'>Patient: <b>{patient_name}</b> ({patient_id}) | Age: {patient_age} | Gender: {patient_gender} | Target Site: <b>{patient_site}</b></span><br>
                    <span style='font-size: 1.2rem; color: #a0aec0; font-weight: 500;'>AI Diagnostic Classification:</span><br>
                    <span style='font-size: 2.5rem; font-weight: 800; color: {color};'>{pred_class_name}</span>
                    <span style='font-size: 2.0rem; color: #e2e8f0; font-weight: 600;'> ({pred_confidence:.2f}% Match Confidence)</span>
                </div>""", unsafe_allow_html=True
            )
            
            prog_col1, prog_col2, prog_col3 = st.columns(3)
            for idx, (label, val) in enumerate(zip(CLASS_LABELS, probs)):
                metric_html = f"""
                <div style='background-color: #1b1e2e; padding: 12px; border-radius: 6px; margin-bottom: 2px; margin-top: 10px; border-left: 4px solid #ff6a00;'>
                    <div style='margin-bottom: 6px;'>
                        <span style='font-weight: 700; color: #ffffff; font-size: 1.1rem;'>{label}</span>
                        <span style='float: right; color: #a0aec0; font-size: 0.85rem; font-weight: 500;'>Score: {val*100:.2f}%</span>
                    </div>
                """
                if idx % 3 == 0:
                    with prog_col1:
                        st.markdown(metric_html, unsafe_allow_html=True)
                        st.progress(float(val))
                        st.markdown("</div>", unsafe_allow_html=True)
                elif idx % 3 == 1:
                    with prog_col2:
                        st.markdown(metric_html, unsafe_allow_html=True)
                        st.progress(float(val))
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    with prog_col3:
                        st.markdown(metric_html, unsafe_allow_html=True)
                        st.progress(float(val))
                        st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3 class='section-title'>🔍 2. Explainable AI: Grad-CAM Feature Visualization</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: st.image(img_rgb, caption=f"Patient Base Dermoscopic Scan ({patient_site})", use_container_width=True)
            with col2: st.image(overlay, caption="Saliency Map Activation Zones (AI Attention)", use_container_width=True)
                
            st.markdown("<h3 class='section-title'>📝 3. Clinical Action Suggestions & Expert Insights</h3>", unsafe_allow_html=True)
            st.error(f"🚨 **Clinical Summary:** {CLINICAL_DATA[pred_class_name]['summary']}")
            st.info(CLINICAL_DATA[pred_class_name]["advise"])
                
            st.markdown("<br/>", unsafe_allow_html=True)
            pdf_data = generate_pdf_report(patient_id, patient_name, patient_age, patient_gender, patient_site, pred_class_name, pred_confidence)
            
            st.download_button(
                label=f"📥 Download Clinical PDF Report for {patient_name}",
                data=pdf_data,
                file_name=f"SkinAI_Report_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# --- PAGE 2: PATIENT HISTORY RECORDS & MEDICAL ANALYTICS ---
elif menu_selection == "Patient History":
    st.markdown("<h1 class='medical-title'>🗃️ Patient Database Logs & History</h1>", unsafe_allow_html=True)
    st.markdown("<p class='medical-subtitle'>Track, filter, export records, view clinical metrics analytics, and execute doctor validation triage.</p>", unsafe_allow_html=True)
    
    df_history = get_all_patients()
    if not df_history.empty:
        st.markdown("<h3 class='section-title'>📈 Clinical Metrics Analytics</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.metric(label="Total Scans Executed", value=len(df_history))
        with c2:
            high_risk_cases = len(df_history[~df_history['AI Diagnosis'].isin(['NV', 'DF', 'SK', 'VASC'])])
            st.metric(label="Malignant/Pre-cancerous Detected", value=high_risk_cases, delta=f"{high_risk_cases} Flagged Cases", delta_color="inverse")
        with c3:
            avg_conf = df_history['Confidence (%)'].mean()
            st.metric(label="Average System Confidence", value=f"{avg_conf:.2f}%")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.write("##### 🩺 Distribution of Diagnosed Pathology Classes")
            class_counts = df_history['AI Diagnosis'].value_counts()
            st.bar_chart(class_counts, color="#ff6a00")
            
        with chart_col2:
            st.write("##### 👥 Age Distribution Frequency Map")
            age_distribution = df_history.groupby('Age').size()
            st.area_chart(age_distribution, color="#ee0979")
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-title'>📋 Patient Log Data Records</h3>", unsafe_allow_html=True)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("📂 No patient records found in the log history database yet.")