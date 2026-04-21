import re

import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
from matplotlib import colormaps
from PIL import Image
import sys, os, io
from scipy.fftpack import dct as scipy_dct

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from architecture import ForgeryDetector
from transformers import DistilBertTokenizer

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="SENTINEL — Forgery Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================
# THEME STATE
# =============================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'light_mode' not in st.session_state:
    st.session_state.light_mode = False

# =============================================================
# THEME — Python picks colors, injects hardcoded CSS (no JS needed)
# =============================================================
def get_theme_css(light: bool) -> str:
    if light:
        bg        = "#f8fafc"
        bg2       = "#ffffff"
        bg3       = "#f1f5f9"
        border    = "#e2e8f0"
        border2   = "#cbd5e1"
        text      = "#1e293b"
        text_dim  = "#64748b"
        accent    = "#2563eb"
        accent_d  = "#2563eb14"
        red       = "#dc2626"
        red_d     = "#dc262614"
        green     = "#16a34a"
        green_d   = "#16a34a14"
        amber     = "#d97706"
        amber_d   = "#d9770614"
        purple    = "#9333ea"
        purple_d  = "#9333ea14"
        shadow    = "0 2px 12px rgba(0,0,0,0.08)"
        sb_bg     = "#ffffff"
    else:
        bg        = "#0e1117"
        bg2       = "#161b25"
        bg3       = "#1c2333"
        border    = "#252f42"
        border2   = "#2e3d57"
        text      = "#d4dff0"
        text_dim  = "#546a87"
        accent    = "#3b82f6"
        accent_d  = "#3b82f618"
        red       = "#ef4444"
        red_d     = "#ef444418"
        green     = "#22c55e"
        green_d   = "#22c55e18"
        amber     = "#f59e0b"
        amber_d   = "#f59e0b18"
        purple    = "#a855f7"
        purple_d  = "#a855f718"
        shadow    = "0 4px 20px rgba(0,0,0,0.35)"
        sb_bg     = "#161b25"

    return f"""
<style>
/* 1. Changed import to fetch Quicksand and DM Mono */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Quicksand:wght@300;400;500;600;700&display=swap');

/* CSS variables — all inline style="color:var(--x)" will work */
:root {{
    --bg:        {bg};
    --bg2:       {bg2};
    --bg3:       {bg3};
    --border:    {border};
    --border2:   {border2};
    --text:      {text};
    --text-dim:  {text_dim};
    --accent:    {accent};
    --accent-d:  {accent_d};
    --red:       {red};
    --red-d:     {red_d};
    --green:     {green};
    --green-d:   {green_d};
    --amber:     {amber};
    --amber-d:   {amber_d};
    --purple:    {purple};
    --purple-d:  {purple_d};
    --mono:      'DM Mono', monospace;
    /* 2. Updated the root sans variable */
    --sans:      'Quicksand', sans-serif;
}}

* {{ box-sizing: border-box; }}
/* 3. Applied Quicksand to the global app body */
html, body, .stApp {{ background: {bg} !important; color: {text} !important; font-family: 'Quicksand', sans-serif !important; letter-spacing: 0.3px !important; }}
#MainMenu, footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button,
section[data-testid="stSidebarCollapsedControl"] {{
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}}
.block-container {{ padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }}
[data-testid="stSidebar"] {{ background: {sb_bg} !important; border-right: 1px solid {border} !important; }}
[data-testid="stSidebar"] section > div {{ background: {sb_bg} !important; }}

/* Sidebar text */
[data-testid="stSidebar"] p {{ color: {text} !important; }}
[data-testid="stSidebar"] label {{ color: {text} !important; }}

/* Sidebar selectbox */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background: {bg3} !important;
    border: 2px solid {accent} !important;
    color: {text} !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] span {{ color: {text} !important; font-weight: 600 !important; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {{ fill: {accent} !important; }}

/* Sliders */
[data-testid="stSlider"] [role="slider"] {{ background: {accent} !important; }}
[data-testid="stSlider"] > div > div > div {{ background: {border} !important; }}
[data-testid="stSlider"] > div > div > div > div {{ background: {accent} !important; }}

.s-header {{ display:flex; align-items:center; justify-content:space-between; padding:1.2rem 0 1rem; border-bottom:1px solid {border}; margin-bottom:1.5rem; }}
.s-logo-row {{ display:flex; align-items:center; gap:1rem; }}
.s-logo {{ width:44px; height:44px; background:{accent_d}; border:1.5px solid {accent}; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; }}
.s-title {{ font-size:1.5rem; font-weight:700; color:{text}; letter-spacing:-0.02em; }}
.s-subtitle {{ font-family:'DM Mono',monospace; font-size:0.62rem; color:{text_dim}; margin-top:0.15rem; }}
.s-badge {{ font-family:'DM Mono',monospace; font-size:0.6rem; padding:0.3rem 0.8rem; border-radius:20px; border:1px solid {green}; color:{green}; background:{green_d}; }}

.sec-label {{ font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.12em; color:{text_dim}; text-transform:uppercase; margin-bottom:0.8rem; display:flex; align-items:center; gap:0.6rem; }}
.sec-label::after {{ content:''; flex:1; height:1px; background:{border}; }}

.card-sm {{ background:{bg3}; border:1px solid {border}; border-radius:8px; padding:0.85rem 1rem; }}

.verdict-wrap {{ border-radius:10px; padding:1.3rem; text-align:center; border:1.5px solid; }}
.verdict-authentic {{ border-color:{green};  background:{green_d}; }}
.verdict-ai        {{ border-color:{purple}; background:{purple_d}; }}
.verdict-photoshop {{ border-color:{amber};  background:{amber_d}; }}
.verdict-spliced   {{ border-color:{red};    background:{red_d}; }}
.verdict-manip     {{ border-color:{amber};  background:{amber_d}; }}
.verdict-icon  {{ font-size:1.8rem; margin-bottom:0.3rem; }}
.verdict-label {{ font-size:1.3rem; font-weight:700; }}
.verdict-desc  {{ font-family:'DM Mono',monospace; font-size:0.62rem; color:{text_dim}; margin-top:0.35rem; }}
.vc-green  {{ color:{green}; }}
.vc-purple {{ color:{purple}; }}
.vc-amber  {{ color:{amber}; }}
.vc-red    {{ color:{red}; }}

.pill-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0.6rem; margin:0.8rem 0; }}
.pill {{ background:{bg3}; border:1px solid {border}; border-radius:8px; padding:0.75rem 0.5rem; text-align:center; }}
.pill-val {{ font-family:'DM Mono',monospace; font-size:1.4rem; font-weight:500; line-height:1; }}
.pill-lbl {{ font-family:'DM Mono',monospace; font-size:0.52rem; color:{text_dim}; letter-spacing:0.08em; margin-top:0.3rem; text-transform:uppercase; }}

.layer-row {{ background:{bg3}; border:1px solid {border}; border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.45rem; display:flex; align-items:center; gap:0.85rem; border-left:3px solid {border2}; }}
.layer-row-green  {{ border-left-color:{green}; }}
.layer-row-amber  {{ border-left-color:{amber}; }}
.layer-row-red    {{ border-left-color:{red}; }}
.layer-row-purple {{ border-left-color:{purple}; }}
.layer-row-dim    {{ border-left-color:{border2}; opacity:0.5; }}
.layer-icon  {{ font-size:1.1rem; flex-shrink:0; }}
.layer-info  {{ flex:1; }}
.layer-name  {{ font-size:0.82rem; font-weight:600; color:{text}; }}
.layer-desc  {{ font-family:'DM Mono',monospace; font-size:0.58rem; color:{text_dim}; margin-top:0.1rem; }}
.layer-right {{ text-align:right; }}
.layer-score {{ font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:500; }}
.layer-risk  {{ font-family:'DM Mono',monospace; font-size:0.56rem; letter-spacing:0.06em; }}

.bar-bg   {{ background:{border}; border-radius:3px; height:3px; margin-top:0.4rem; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:3px; }}

[data-testid="stFileUploader"] {{ border:1.5px dashed {border2} !important; border-radius:10px !important; background:{bg2} !important; }}
[data-testid="stFileUploader"]:hover {{ border-color:{accent} !important; }}

/* 4. Applied Quicksand to buttons */
.stButton > button {{ background:{accent} !important; border:none !important; color:#fff !important; font-family:'Quicksand',sans-serif !important; font-size:0.85rem !important; font-weight:600 !important; padding:0.65rem 1.5rem !important; border-radius:8px !important; letter-spacing: 0.5px !important; }}
.stButton > button:hover {{ opacity:0.88 !important; }}
.stButton > button:disabled {{ background:{border2} !important; opacity:0.5 !important; }}

.stTabs [data-baseweb="tab-list"] {{ background:{bg2} !important; border-bottom:1px solid {border} !important; gap:0 !important; }}
.stTabs [data-baseweb="tab"] {{ font-family:'DM Mono',monospace !important; font-size:0.65rem !important; letter-spacing:0.06em !important; color:{text_dim} !important; padding:0.5rem 0.9rem !important; }}
.stTabs [aria-selected="true"] {{ color:{accent} !important; border-bottom:2px solid {accent} !important; background:{accent_d} !important; }}

.sb-row {{ display:flex; justify-content:space-between; align-items:center; padding:0.45rem 0; border-bottom:1px solid {border}; font-family:'DM Mono',monospace; font-size:0.65rem; color:{text_dim}; }}
.sb-val {{ color:{accent}; font-weight:500; }}

textarea {{ background:{bg2} !important; border:1px solid {border2} !important; color:{text} !important; font-family:'DM Mono',monospace !important; font-size:0.78rem !important; border-radius:8px !important; }}

.empty-state {{ border:1.5px dashed {border2}; border-radius:10px; padding:3rem 2rem; text-align:center; }}
.empty-icon  {{ font-size:2.2rem; opacity:0.3; }}
.empty-text  {{ font-family:'DM Mono',monospace; font-size:0.7rem; color:{text_dim}; margin-top:0.8rem; line-height:1.9; }}

.info-row {{ background:{bg3}; border:1px solid {border}; border-radius:8px; padding:0.8rem 1rem; font-family:'DM Mono',monospace; font-size:0.63rem; color:{text_dim}; line-height:2; }}
.iv {{ color:{text}; }}

.rbox {{ border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.45rem; border:1px solid {border}; }}
.rbox-real    {{ border-left:3px solid {green};   background:{green_d}; }}
.rbox-fake    {{ border-left:3px solid {red};     background:{red_d}; }}
.rbox-ai      {{ border-left:3px solid {purple};  background:{purple_d}; }}
.rbox-neutral {{ border-left:3px solid {border2}; background:{bg3}; }}

.stat-card {{ background:{bg2}; border:1px solid {border}; border-top:2px solid {accent}; border-radius:8px; padding:1rem 0.6rem; text-align:center; }}
.stat-val  {{ font-family:'DM Mono',monospace; font-size:1.25rem; font-weight:500; color:{accent}; }}
.stat-lbl  {{ font-family:'DM Mono',monospace; font-size:0.52rem; color:{text_dim}; letter-spacing:0.08em; margin-top:0.25rem; text-transform:uppercase; }}

.stSpinner > div {{ border-top-color:{accent} !important; }}
</style>
"""

# Inject theme CSS immediately — Python decides the colors, no JS needed
is_light = (st.session_state.get('theme', 'dark') == 'light')
st.markdown(get_theme_css(is_light), unsafe_allow_html=True)


# =============================================================
# DETECTION FUNCTIONS
# =============================================================

def dct_frequency_analysis(image_pil):
    img_gray    = np.array(image_pil.convert('L')).astype(np.float32)
    img_gray    = cv2.resize(img_gray, (256, 256))
    dct_coeffs  = scipy_dct(scipy_dct(img_gray, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_log     = np.log(np.abs(dct_coeffs) + 1e-8)
    h, w        = dct_log.shape
    hf_energy   = np.mean(np.abs(dct_log[h//2:, w//2:]))
    lf_energy   = np.mean(np.abs(dct_log[:h//4, :w//4]))
    hf_ratio    = hf_energy / (lf_energy + 1e-8)
    dct_abs     = np.abs(dct_coeffs)
    peak_ratio  = np.sum(dct_abs > np.percentile(dct_abs, 99)) / dct_abs.size
    noise_floor = np.std(dct_log[h//2:, w//2:])
    hf_score    = max(0, min(1, 1 - (hf_ratio / 0.15)))
    peak_score  = max(0, min(1, peak_ratio / 0.005))
    noise_score = max(0, min(1, 1 - (noise_floor / 2.5)))
    ai_score    = 0.45*hf_score + 0.35*peak_score + 0.20*noise_score
    smoothness  = 1.0 / (np.std(np.array(image_pil.convert('RGB')).astype(np.float32)) / 50.0 + 1.0)
    ai_score    = min(1.0, ai_score + 0.15 * smoothness)
    # Photoshop/edit: 8x8 JPEG block artifact detection
    img_arr     = np.array(image_pil.convert('L').resize((256, 256))).astype(np.float32)
    horiz       = np.mean(np.abs(np.diff(img_arr, axis=0)[7::8, :]))
    overall     = np.mean(np.abs(np.diff(img_arr, axis=0)))
    block_ratio = horiz / (overall + 1e-8)
    ps_score    = max(0.0, min(1.0, (block_ratio - 1.0) / 3.0))
    return float(ai_score), float(ps_score), {
        'hf_ratio': float(hf_ratio), 'peak_ratio': float(peak_ratio),
        'noise_floor': float(noise_floor), 'block_ratio': float(block_ratio)
    }


def ela_analysis(image_pil):
    """Error Level Analysis — re-compress at known quality, amplify diff. Edits show bright."""
    buf = io.BytesIO()
    image_pil.convert('RGB').save(buf, format='JPEG', quality=75)
    buf.seek(0)
    recomp = Image.open(buf).convert('RGB')
    orig   = np.array(image_pil.convert('RGB')).astype(np.float32)
    rc     = np.array(recomp.resize((image_pil.width, image_pil.height))).astype(np.float32)
    diff   = np.abs(orig - rc)
    ela_map = diff / (diff.max() + 1e-8)
    ela_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
    high_pct = float(np.mean(ela_map > 0.4))
    ela_std  = float(np.std(ela_map))
    score    = min(1.0, high_pct * 3.0 + ela_std * 2.0)
    ela_gray = np.mean(ela_map, axis=2)
    return float(score), ela_gray, ela_vis


def screenshot_detection(image_pil):
    """
    Detects screenshots and computer-generated images (not AI, not photos).
    Signals: near-zero noise, dominant H/V edges, no chromatic aberration,
    sparse color palette, no JPEG-style sensor grain.
    Returns 0.0-1.0 where 1.0 = almost certainly a screenshot/UI image.
    """
    try:
        img_np   = np.array(image_pil.convert('RGB')).astype(np.float32)
        img_gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w     = img_gray.shape

        # Signal 1: Noise level — screenshots have near-zero sensor noise
        blurred   = cv2.GaussianBlur(img_gray, (3, 3), 0)
        mean_noise = float(np.mean(np.abs(img_gray.astype(np.float32) - blurred.astype(np.float32))))
        noise_score = max(0.0, min(1.0, 1.0 - mean_noise / 0.8))

        # Signal 2: Horizontal/vertical edge dominance — UI has straight grid lines
        edges  = cv2.Canny(img_gray, 50, 150)
        kh     = np.ones((1, 25), np.uint8)
        kv     = np.ones((25, 1), np.uint8)
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kh)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kv)
        total_e = float(np.sum(edges > 0)) + 1e-8
        hv_ratio = float(np.sum(h_lines > 0) + np.sum(v_lines > 0)) / total_e
        hv_score = max(0.0, min(1.0, (hv_ratio - 0.10) / 0.50))

        # Signal 3: No chromatic aberration — cameras have R/B channel edge offset,
        # screenshots have perfect pixel-aligned channels
        r_edges = cv2.Canny(img_np[:, :, 0].astype(np.uint8), 30, 100)
        b_edges = cv2.Canny(img_np[:, :, 2].astype(np.uint8), 30, 100)
        aberration = float(np.mean(np.logical_xor(r_edges > 0, b_edges > 0)))
        aber_score = max(0.0, min(1.0, 1.0 - aberration / 0.02))

        # Signal 4: Sparse color palette — UI uses brand colors not continuous gradients
        # Count distinct color bins with >0.1% of pixels across R, G, B
        hist_bins = []
        for ch in range(3):
            hist = cv2.calcHist([img_np[:, :, ch].astype(np.uint8)], [0], None, [256], [0, 256])
            hist_bins.append(float(np.sum(hist.flatten() / hist.sum() > 0.001)))
        mean_bins  = float(np.mean(hist_bins))
        palette_score = max(0.0, min(1.0, 1.0 - (mean_bins - 20.0) / 130.0))

        # Signal 5: Large uniform flat regions — UI backgrounds, panels
        block_size = 16
        flat_count = 0
        total_b    = 0
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img_gray[y:y+block_size, x:x+block_size].astype(np.float32)
                if np.std(block) < 2.0:
                    flat_count += 1
                total_b += 1
        flat_ratio = flat_count / (total_b + 1e-8)
        flat_score = max(0.0, min(1.0, (flat_ratio - 0.15) / 0.50))

        score = (0.30 * noise_score + 0.25 * hv_score + 0.20 * aber_score +
                 0.15 * palette_score + 0.10 * flat_score)
        return float(score), {
            'noise': mean_noise, 'hv_ratio': hv_ratio,
            'aberration': aberration, 'mean_bins': mean_bins, 'flat_ratio': flat_ratio
        }
    except Exception:
        return 0.0, {}


def photoshop_detection(image_pil):
    """
    Detects Photoshop edits, compositing and local manipulation.
    Signals: ELA regional inconsistency (edited regions recompress differently),
    copy-move detection (duplicated patches), and noise inconsistency
    between image regions (spliced patches have different noise profiles).
    Returns 0.0-1.0 where 1.0 = strong evidence of post-processing.
    """
    try:
        img_rgb  = image_pil.convert('RGB')
        img_np   = np.array(img_rgb).astype(np.float32)
        img_gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w     = img_gray.shape

        # Signal 1: ELA regional inconsistency
        # Authentic images: all regions recompress similarly → low variance
        # Edited images: pasted regions have different compression history → high variance
        buf = io.BytesIO()
        img_rgb.save(buf, 'JPEG', quality=75)
        buf.seek(0)
        recomp   = np.array(Image.open(buf).convert('RGB')).astype(np.float32)
        ela      = np.abs(img_np - recomp).mean(axis=2)
        # Divide into 4x4 grid, measure ELA mean per region
        rh, rw   = h // 4, w // 4
        r_means  = []
        for ry in range(4):
            for rx in range(4):
                r = ela[ry*rh:(ry+1)*rh, rx*rw:(rx+1)*rw]
                r_means.append(float(np.mean(r)))
        ela_range     = float(max(r_means) - min(r_means))
        ela_reg_score = max(0.0, min(1.0, (ela_range - 3.0) / 15.0))

        # Also check: multi-quality resave (quality 90 vs 75) — detects double-compression
        buf2 = io.BytesIO()
        img_rgb.save(buf2, 'JPEG', quality=90)
        buf2.seek(0)
        recomp2  = np.array(Image.open(buf2).convert('RGB')).astype(np.float32)
        ela2     = np.abs(img_np - recomp2).mean(axis=2)
        r2_means = []
        for ry in range(4):
            for rx in range(4):
                r = ela2[ry*rh:(ry+1)*rh, rx*rw:(rx+1)*rw]
                r2_means.append(float(np.mean(r)))
        ela2_range     = float(max(r2_means) - min(r2_means))
        ela2_reg_score = max(0.0, min(1.0, (ela2_range - 2.0) / 10.0))

        # Signal 2: Noise inconsistency between regions
        # Authentic: uniform noise level throughout. Spliced: pasted area has different noise.
        blurred    = cv2.GaussianBlur(img_gray, (5, 5), 0)
        noise_map  = np.abs(img_gray.astype(np.float32) - blurred.astype(np.float32))
        rh2, rw2   = h // 8, w // 8
        noise_levels = []
        for ry in range(8):
            for rx in range(8):
                region = noise_map[ry*rh2:(ry+1)*rh2, rx*rw2:(rx+1)*rw2]
                noise_levels.append(float(np.mean(region)))
        noise_cv      = float(np.std(noise_levels) / (np.mean(noise_levels) + 1e-8))
        noise_incon   = max(0.0, min(1.0, (noise_cv - 0.20) / 0.60))

        # Signal 3: Copy-move detection (simplified block matching)
        # Resize to manageable size, extract overlapping blocks, find near-duplicates
        small  = cv2.resize(img_gray, (128, 128)).astype(np.float32)
        bs     = 16
        blocks = {}
        copy_hits = 0
        for y in range(0, 128 - bs, bs // 2):
            for x in range(0, 128 - bs, bs // 2):
                block  = small[y:y+bs, x:x+bs]
                key    = tuple((block[::4, ::4] / 32).astype(int).flatten())
                if key in blocks:
                    dy = abs(y - blocks[key][0])
                    dx = abs(x - blocks[key][1])
                    if dy > bs or dx > bs:  # not just adjacent blocks
                        copy_hits += 1
                else:
                    blocks[key] = (y, x)
        total_blocks  = len(blocks)
        copy_score    = max(0.0, min(1.0, copy_hits / (total_blocks * 0.05 + 1e-8)))

        score = (0.40 * ela_reg_score + 0.20 * ela2_reg_score +
                 0.25 * noise_incon   + 0.15 * copy_score)
        return float(score), {
            'ela_range': ela_range, 'ela2_range': ela2_range,
            'noise_cv': noise_cv, 'copy_hits': copy_hits
        }
    except Exception:
        return 0.0, {}


def metadata_analysis(image_pil):
    """
    Analyse image metadata for AI/edit signals.

    Real camera photos:  have EXIF with Make/Model/ISO/FNumber/ExposureTime
    AI generated images: almost never have EXIF; often output at square power-of-2 sizes
    Photoshop edited:    may have Software tag = Adobe Photoshop / GIMP / Lightroom
    WhatsApp/social:     strips EXIF but keeps dimensions → ambiguous

    Returns score in [-0.40, 0.80]:
      positive = AI/edited evidence
      negative = real camera evidence (lowers final fake score)
    """
    try:
        from PIL.ExifTags import TAGS
        w, h = image_pil.size

        score    = 0.0
        signals  = {}

        # ── EXIF ──────────────────────────────────────────────────────────────
        try:
            exif_data = image_pil._getexif()
        except Exception:
            exif_data = None

        if exif_data is None:
            score += 0.15                    # no EXIF: moderate AI signal
            signals['exif'] = 'absent'
        else:
            tags = {TAGS.get(tid, str(tid)): str(v)[:80]
                    for tid, v in exif_data.items()}

            camera_fields = {'Make','Model','ExposureTime','FNumber',
                             'ISOSpeedRatings','FocalLength'}
            cam_hits = sum(1 for f in camera_fields if f in tags)

            if cam_hits >= 3:
                score -= 0.30               # strong real camera evidence
                signals['exif'] = f'camera ({cam_hits} fields)'
            elif cam_hits > 0:
                score -= 0.12
                signals['exif'] = f'partial camera ({cam_hits} fields)'
            else:
                score += 0.08               # EXIF present but no camera fields
                signals['exif'] = 'stripped'

            sw = tags.get('Software', '').lower()
            if any(x in sw for x in ['midjourney','stable diffusion','dall-e','firefly','imagen','runway']):
                score += 0.65
                signals['software'] = tags.get('Software','')
            elif any(x in sw for x in ['photoshop','gimp','lightroom','affinity','canva']):
                score += 0.35
                signals['software'] = tags.get('Software','')

        # ── Dimensions ────────────────────────────────────────────────────────
        ratio    = w / h
        ai_sizes = {512, 640, 768, 832, 1024, 1152, 1280, 1344, 1536, 2048}
        is_square = abs(ratio - 1.0) < 0.02
        both_ai   = w in ai_sizes and h in ai_sizes
        is_camera = (abs(ratio - 4/3) < 0.03 or abs(ratio - 3/2) < 0.03 or
                     abs(ratio - 3/4) < 0.03 or abs(ratio - 2/3) < 0.03 or
                     abs(ratio - 16/9) < 0.03)

        if is_square and both_ai:
            score += 0.35
            signals['dimensions'] = f'{w}x{h} (AI square std)'
        elif is_square:
            score += 0.22
            signals['dimensions'] = f'{w}x{h} (AI square)'
        elif both_ai:
            score += 0.18
            signals['dimensions'] = f'{w}x{h} (AI size)'
        elif is_camera and exif_data is not None:
            score -= 0.10
            signals['dimensions'] = f'{w}x{h} (camera ratio+exif)'
        else:
            signals['dimensions'] = f'{w}x{h}'

        # ── Format ────────────────────────────────────────────────────────────
        if getattr(image_pil, 'format', None) == 'PNG':
            score += 0.08
            signals['format'] = 'PNG'

        score = float(max(-0.40, min(0.80, score)))
        signals['score'] = round(score, 3)
        return score, signals

    except Exception:
        return 0.0, {}




@st.cache_resource
def load_clip_model():
    try:
        from transformers import CLIPProcessor, CLIPModel
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        m.eval()
        return m, p, True
    except Exception:
        return None, None, False


def visual_ai_detection(image_pil):
    """
    Pure numpy/cv2 AI image detector — no torch, never crashes.
    Calibrated on 3 AI image styles: vivid portrait, B&W CGI, soft photorealistic.
    Returns score 0.0-1.0 where 1.0 = definitely AI generated.
    """
    try:
        img_np    = np.array(image_pil.convert('RGB')).astype(np.float32)
        img_gray  = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        img_hsv   = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        mean_sat  = float(np.mean(img_hsv[:, :, 1]))
        is_monochrome = mean_sat < 50

        # HF ratio via DCT — AI generators produce low high-freq content
        # Calibrated: AI 0.008-0.035, real photos 0.050-0.120
        from scipy.fftpack import dct as _dct
        _g256    = cv2.resize(img_gray, (256, 256)).astype(np.float32)
        _dct2d   = _dct(_dct(_g256, axis=0, norm='ortho'), axis=1, norm='ortho')
        _da      = np.abs(_dct2d)
        hf_ratio = float(_da[64:, 64:].mean() / (_da[:32, :32].mean() + 1e-8))
        hf_score = max(0.0, min(1.0, 1.0 - (hf_ratio - 0.008) / 0.052))

        # Bilateral residual — measures grain/noise level
        _bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
        bilateral_residual = float(np.mean(
            np.abs(img_gray - _bilateral.astype(np.float32))
        ))

        if is_monochrome:
            # B&W / CGI / metallic AI art path
            bright_pct    = float(np.mean(img_gray > 240))
            dark_pct      = float(np.mean(img_gray < 15))
            extreme_score = max(0.0, min(1.0, (bright_pct + dark_pct - 0.05) / 0.25))
            _r = img_np[:, :, 0].flatten()[:8000]
            _g = img_np[:, :, 1].flatten()[:8000]
            _b = img_np[:, :, 2].flatten()[:8000]
            rg_corr    = float(np.corrcoef(_r, _g)[0, 1])
            rb_corr    = float(np.corrcoef(_r, _b)[0, 1])
            mono_score = max(0.0, min(1.0, (max(rg_corr, rb_corr) - 0.90) / 0.08))
            bilateral_s = max(0.0, min(1.0, 1.0 - (bilateral_residual - 1.0) / 5.0))
            return (0.40 * mono_score + 0.35 * extreme_score +
                    0.15 * hf_score   + 0.10 * bilateral_s)
        else:
            # Color AI path — low HF, oversaturated, smooth
            sat_score   = max(0.0, min(1.0, (mean_sat - 70.0) / 100.0))
            bilateral_s = max(0.0, min(1.0, 1.0 - (bilateral_residual - 1.0) / 5.0))
            visual = 0.65 * hf_score + 0.20 * sat_score + 0.15 * bilateral_s
            if hf_ratio < 0.020:
                visual = max(visual, 0.72)
            return visual
    except Exception:
        return 0.5


def real_photo_confidence(image_pil):
    """
    Estimates how likely this is a genuine real photo (not AI/edited).
    Returns 0.0-1.0 where high = strong real photo evidence.
    Used to SUPPRESS the AI track when real photo signals are strong.
    
    Key signals that survive JPEG compression:
    - Phone portrait/landscape aspect ratio (9:16, 3:4)
    - Non-square, non-AI-standard dimensions
    - Chromatic aberration (camera lens property, absent in AI)
    - High local contrast variance (depth of field = real optics)
    """
    try:
        w, h    = image_pil.size
        ratio   = w / h
        img_np  = np.array(image_pil.convert('RGB')).astype(np.float32)
        img_gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        score    = 0.0
        ai_sizes = {512, 640, 768, 832, 1024, 1152, 1280, 1344, 1536, 2048}
        is_square = abs(ratio - 1.0) < 0.02

        # Phone portrait/landscape — 9:16 (0.5625), 3:4 (0.75), 4:3, 16:9
        is_phone_portrait  = 0.45 < ratio < 0.67
        is_phone_landscape = 1.48 < ratio < 2.25
        if is_phone_portrait or is_phone_landscape:
            score += 0.35   # very strong real signal

        # Non-square AND non-AI-standard dimensions
        not_ai_dims = not is_square and (w not in ai_sizes) and (h not in ai_sizes)
        if not_ai_dims:
            score += 0.20

        # Chromatic aberration — camera lenses produce R/B channel edge offset
        # AI images don't have this (generated pixel-perfect)
        try:
            r_edges = cv2.Canny(img_np[:, :, 0].astype(np.uint8), 30, 100)
            b_edges = cv2.Canny(img_np[:, :, 2].astype(np.uint8), 30, 100)
            aberration = float(np.mean(np.logical_xor(r_edges > 0, b_edges > 0)))
            if aberration > 0.018:
                score += 0.20   # real camera chromatic aberration present
        except Exception:
            pass

        # Local contrast variance (depth of field)
        # Real photos: high CV because some regions sharp, some blurry
        # AI images: more uniform sharpness throughout
        try:
            bsz = 32
            lcs = []
            hh, ww = img_gray.shape
            for y in range(0, hh - bsz, bsz):
                for x in range(0, ww - bsz, bsz):
                    block = img_gray[y:y+bsz, x:x+bsz].astype(np.float32)
                    lcs.append(float(np.std(block)))
            if lcs:
                lc_cv = float(np.var(lcs)) / (float(np.mean(lcs))**2 + 1e-8)
                if lc_cv > 0.60:
                    score += 0.15
        except Exception:
            pass

        return float(min(1.0, score))
    except Exception:
        return 0.0


def clip_ai_detection(image_pil, cm, cp):
    """CLIP-only semantic detector. Returns 0.5 on any failure."""
    try:
        import torch.nn.functional as F_

        real_prompts = [
            "a photograph", "a photo taken by a camera", "a real life photo",
            "DSLR photo", "phone camera photo", "candid photograph",
            "photojournalism", "documentary photo",
        ]
        ai_prompts = [
            "digital art", "AI generated art", "concept art",
            "highly detailed digital painting", "artstation trending",
            "midjourney", "stable diffusion", "CGI render",
            "hyper realistic digital art", "synthetic image",
            "cinematic AI photo", "AI generated portrait",
            "perfect lighting studio photography",
            "3D render unreal engine", "photorealistic render",
        ]

        img_inputs = cp(images=image_pil, return_tensors="pt")
        with torch.no_grad():
            img_feat = F_.normalize(cm.get_image_features(**img_inputs), dim=-1)

        def get_centroid(prompts):
            enc = cp(text=prompts, return_tensors="pt", padding=True)
            with torch.no_grad():
                tf = F_.normalize(cm.get_text_features(**enc), dim=-1)
            return tf.mean(dim=0, keepdim=True)

        real_c = get_centroid(real_prompts)
        ai_c   = get_centroid(ai_prompts)

        sim_real = torch.mm(img_feat, real_c.T).item()
        sim_ai   = torch.mm(img_feat, ai_c.T).item()
        temp     = 0.07
        probs    = torch.softmax(torch.tensor([sim_real / temp, sim_ai / temp]), dim=0)
        return float(probs[1].item()), True

    except Exception:
        return 0.5, False


def detect_ai_caption(caption, cm, cp, clip_ok):
    """
    Detect if a caption was written by AI.
    Covers three AI caption styles:
    - Verbose descriptor: "a stunning warrior bathed in golden light..."
    - Poetic slogan: "Polished outside. Something else inside."
    - Abstract metaphor: "Where the ocean meets the sky, time forgets itself."
    """
    if not caption or not caption.strip():
        return None
    text = caption.strip()
    tl   = text.lower()
    words = tl.split()
    wc    = len(words)
    clean_words = [w.strip('.,!?;:\'")') for w in words]

    # ── Signal 1: AI descriptor phrases (verbose style) ──────────────────────
    ai_phrases = [
        'showcasing','depicting','highlighting','featuring',
        'in this image','in the image','the image shows','the image depicts',
        'illustrating','demonstrating','capturing the essence',
        'a stunning','a breathtaking','this photograph','this photo shows',
        'set against','bathed in','surrounded by','adorned with',
        'a powerful','an epic','dramatic','ethereal','majestic',
        'cinematic','dynamic','vibrant','intricate',
        'golden light','soft light','warm light','glowing','radiant',
        'warrior','guardian','serene','mysterious','haunting',
        'breathtaking','mesmerizing','captivating','awe-inspiring',
    ]
    phrase_hits = sum(1 for p in ai_phrases if p in tl)
    phrase_score = min(0.60, phrase_hits * 0.10)  # 6+ hits → 0.60, not capped at 0.15

    # ── Signal 2: Poetic/slogan structure ─────────────────────────────────────
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    verbs = {'is','are','was','were','has','have','had','do','does','did',
             'will','would','can','could','should','might','must',
             'runs','walks','stands','sits','looks','feels','seems','appears',
             'meets','forgets','finds','seeks','holds','breaks','fades',
             'rises','falls','flows','grows','shines','burns','turns',
             'took','went','came','saw','got','made','said'}
    has_verb     = any(w in verbs for w in clean_words)
    parallel     = len(sentences) >= 2 and all(len(s.split()) <= 7 for s in sentences)
    frag_style   = parallel and not has_verb    # verbless balanced fragments = very AI
    # Single-sentence poetic: short, subjectless, abstract (e.g. "Where silence meets steel.")
    first_word    = clean_words[0] if clean_words else ''
    subjectless   = first_word not in {'i','my','a','an','the','this','that','here','we','it','he','she'}
    poetic_single = (len(sentences) == 1 and wc <= 7 and subjectless)

    # ── Signal 3: Abstract/poetic vocabulary ─────────────────────────────────
    abstract_words = {
        'soul','essence','beauty','power','strength','truth','light','shadow',
        'silence','chaos','order','freedom','hope','fear','spirit','heart',
        'void','echo','pulse','depth','grace','rage','calm','storm',
        'stillness','darkness','horizon','journey','destiny','legacy',
        'ocean','sky','earth','fire','wind','sun','moon','star',
        'memory','eternity','infinity','beginning','reality','illusion',
        'mystery','wonder','glory','dream','time','space',
        # Added: common poetic AI single-word abstracts
        'steel','iron','gold','silver','shadow','shadows','whisper','whispers',
        'polished','raw','broken','shattered','forged','carved','woven',
        'boundaries','limits','borders','edges','walls','chains',
        'outside','inside','within','beneath','above','beyond',
        'rise','fall','burn','fade','breathe','flow','bloom','decay',
        'everything','nothing','something','anything','somewhere','nowhere',
    }
    abstract_hits = sum(1 for w in clean_words if w in abstract_words)

    # ── Signal 4: Poetic metaphor constructs ──────────────────────────────────
    contrast  = bool(re.search(r'\b(but|yet|however|though|while|outside|inside|beneath|beyond|within)\b', tl))
    poetic    = bool(re.search(r'\b(where|when|as|like|becomes|whispers|speaks|breathes|dances|watches|forgets|meets)\b', tl))
    reflexive = bool(re.search(r'\b(itself|himself|herself|themselves)\b', tl))
    # Comma-balanced: two halves of similar word count
    comma_balanced = False
    if ',' in text:
        parts = text.split(',', 1)
        ratio = len(parts[0].split()) / (len(parts[1].split()) + 1e-8)
        comma_balanced = 0.4 < ratio < 2.5 and abstract_hits > 0

    # ── Signal 5: Average word length (AI uses elevated vocabulary) ───────────
    avg_wl = float(np.mean([len(w) for w in clean_words if w])) if clean_words else 0
    vocab_score = max(0.0, min(0.10, (avg_wl - 4.2) / 4.0))

    # ── Combine ───────────────────────────────────────────────────────────────
    score = 0.0
    score += phrase_score                                    # verbose AI phrases
    score += 0.25 if frag_style else 0.0                   # verbless parallel fragments
    score += 0.12 if parallel and not frag_style else 0.0  # parallel with verb still AI
    score += 0.20 if poetic_single else 0.0                # single short subjectless poetic
    score += 0.10 if contrast else 0.0                     # contrast/juxtaposition
    score += min(0.18, abstract_hits * 0.06)               # abstract vocab
    score += 0.08 if poetic else 0.0                       # poetic metaphor
    score += 0.05 if reflexive else 0.0
    score += 0.05 if comma_balanced else 0.0
    score += vocab_score
    # Short verbless fragments with no other signals still lean AI
    score += 0.08 if not has_verb and wc <= 6 and wc >= 3 else 0.0

    score = max(0.0, min(1.0, score))
    label = 'AI-GENERATED' if score > 0.42 else 'HUMAN-WRITTEN'
    conf  = score if score > 0.42 else (1.0 - score)
    return {
        'score': score, 'label': label, 'confidence': conf,
        'phrase_hits': phrase_hits, 'word_count': wc,
        'heuristic': score, 'clip_signal': 0.5,
        'signals': {
            'phrase_score': round(phrase_score, 2),
            'frag_style': frag_style,
            'parallel': parallel,
            'abstract_hits': abstract_hits,
            'contrast': contrast,
            'poetic': poetic,
        }
    }


@st.cache_resource
def load_sbi_model():
    device  = torch.device('cpu')
    model   = ForgeryDetector()
    ckpt    = torch.load('best_model.pth', map_location=device)

    # Robust key detection — handle both checkpoint formats
    if isinstance(ckpt, dict):
        # Format 1: {'model_state': ..., 'epoch': ..., 'val_f1': ...}
        if 'model_state' in ckpt:
            state = ckpt['model_state']
        # Format 2: {'state_dict': ...}
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        # Format 3: bare state dict (keys start with layer names)
        elif any(k.startswith(('vision_branch','text_branch','fusion','unet')) for k in ckpt):
            state = ckpt
        else:
            # Try first value that is itself a dict
            state = next((v for v in ckpt.values() if isinstance(v, dict)), ckpt)
    else:
        state = ckpt  # raw OrderedDict

    model.load_state_dict(state, strict=True)
    model.eval()
    tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tok, device


# def run_sbi_inference(image_pil, caption, model, tok, device):
#     """
#     FIX 2 — Image preprocessing:
#     - Resize PIL first, then convert to tensor (avoids cv2 BGR issue on some systems)
#     - Keep uint8 numpy for display, use PIL path for transforms (matches training)
#     - Clamp caption to non-empty string (empty string causes DistilBERT warning)
#     """
#     IMG = 224

#     # Step 1: Resize using PIL (same as training augmentation pipeline)
#     img_resized = image_pil.convert('RGB').resize((IMG, IMG), Image.BILINEAR)

#     # Step 2: Keep uint8 numpy copy for display / region highlighting
#     image_np = np.array(img_resized)  # HWC uint8 RGB

#     # Step 3: Apply same normalization as training (ImageNet stats)
#     tf    = T.Compose([
#         T.ToTensor(),                                      # [0,255] HWC → [0,1] CHW
#         T.Normalize([0.485, 0.456, 0.406],
#                     [0.229, 0.224, 0.225])
#     ])
#     img_t = tf(img_resized).unsqueeze(0).to(device)        # (1,3,224,224)

#     # Step 4: Normalize caption to match training dataset format (Flickr8k style)
#     # Training captions were: lowercase, stripped, no excessive punctuation
#     import re as _re
#     if caption and caption.strip():
#         cap_text = caption.strip().lower()
#         cap_text = _re.sub(r"[^a-z0-9 .,!?'-]", " ", cap_text)  # remove special chars
#         cap_text = _re.sub(r"\s+", " ", cap_text).strip()        # normalize whitespace
#         if not cap_text:
#             cap_text = "an image"
#     else:
#         cap_text = "an image"

#     enc = tok(
#         cap_text,
#         padding='max_length',
#         max_length=128,
#         truncation=True,
#         return_tensors='pt'
#     )

#     # Step 5: Forward pass
#     with torch.no_grad():
#         logit, heatmap = model(
#             img_t,
#             enc['input_ids'].to(device),
#             enc['attention_mask'].to(device)
#         )

#     # Step 6: sigmoid on logit → probability in [0,1]
#     # Model was trained with BCEWithLogitsLoss → logit > 0 = fake, < 0 = real
#     prob     = torch.sigmoid(logit).item()
#     heat_np  = heatmap.squeeze().cpu().numpy()   # (224,224) float32 in [0,1]

#     return prob, heat_np, image_np

def run_sbi_inference(image_pil, caption, model, tok, device):
    IMG = 224

    # Step 1: Resize for the AI model
    img_resized = image_pil.convert('RGB').resize((IMG, IMG), Image.BILINEAR)

    # Step 2: Grab the FULL RESOLUTION image for the UI
    full_res_np = np.array(image_pil.convert('RGB')) 

    # Step 3: Apply normalization
    tf    = T.Compose([
        T.ToTensor(),                                      
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    img_t = tf(img_resized).unsqueeze(0).to(device)        

    # Step 4: Normalize caption
    import re as _re
    if caption and caption.strip():
        cap_text = caption.strip().lower()
        cap_text = _re.sub(r"[^a-z0-9 .,!?'-]", " ", cap_text) 
        cap_text = _re.sub(r"\s+", " ", cap_text).strip()       
        if not cap_text: cap_text = "an image"
    else:
        cap_text = "an image"

    enc = tok(cap_text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')

    # Step 5: Forward pass
    with torch.no_grad():
        logit, heatmap = model(img_t, enc['input_ids'].to(device), enc['attention_mask'].to(device))

    prob     = torch.sigmoid(logit).item()
    heat_np  = heatmap.squeeze().cpu().numpy()  

    # RETURN THE FULL RES IMAGE
    return prob, heat_np, full_res_np

def segment_regions(heatmap_np, image_np, threshold=0.5):
    """
    FIX 5 — Improved region highlighting.

    Changes:
    - Smooth heatmap with Gaussian blur before thresholding (removes salt/pepper noise)
    - Morphological closing fills small holes in fake mask
    - Soft alpha blend using heatmap intensity (not hard binary) for smoother edges
    - Real regions dimmed more aggressively (better visual contrast)
    - Contour min area raised to 200px (removes tiny false-positive specks)
    - Red channel boost on fake regions instead of flat red overlay
    """
    h, w = image_np.shape[:2]

    # Step 1: Upsample heatmap to image size
    hm = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_CUBIC)

    # Step 2: Gaussian blur to reduce noise
    hm_smooth = cv2.GaussianBlur(hm, (15, 15), 0)

    # Step 3: Hard mask from smoothed heatmap
    fake_mask = (hm_smooth > threshold).astype(np.uint8)

    # Step 4: Morphological closing — fill gaps inside detected regions
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fake_mask = cv2.morphologyEx(fake_mask, cv2.MORPH_CLOSE, kernel)

    real_mask = 1 - fake_mask

    # Step 5: Soft blend using heatmap intensity for fake regions
    # (instead of hard color flip — gives gradual highlighting at edges)
    alpha_map  = np.clip(hm_smooth, 0, 1)[:, :, None]   # (H,W,1)
    gray_img   = np.stack([cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)] * 3, axis=2)

    # Real regions: strong desaturation (30% color, 70% gray)
    real_part  = (real_mask[:, :, None] *
                  (0.30 * image_np.astype(np.float32) +
                   0.70 * gray_img.astype(np.float32)))

    # Fake regions: full color + red channel boost proportional to heatmap
    fake_part  = fake_mask[:, :, None] * image_np.astype(np.float32)
    red_boost  = np.zeros_like(fake_part)
    red_boost[:, :, 0] = 255
    fake_part  = fake_part + fake_mask[:, :, None] * alpha_map * red_boost * 0.45

    out = np.clip(real_part + fake_part, 0, 255).astype(np.uint8)

    # Step 6: Draw contours — only significant regions
    contours, _ = cv2.findContours(fake_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_contours = [c for c in contours if cv2.contourArea(c) > 200]
    cv2.drawContours(out, big_contours, -1, (255, 50, 50), 2)

    fake_pct  = float(fake_mask.mean() * 100)
    n_regions = len(big_contours)
    return out, fake_pct, 100.0 - fake_pct, n_regions


def create_overlay(image_np, heatmap_np, alpha=0.45):
    # 1. Get the dimensions of the full-res image
    h, w = image_np.shape[:2]
    
    # 2. Stretch the 224x224 heatmap to perfectly match the full image
    hm_resized = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Apply the jet color map to the newly resized heatmap
    colored = colormaps['jet'](hm_resized)[:, :, :3]
    
    # 4. Blend them smoothly together
    overlay = alpha * colored + (1-alpha) * (image_np / 255.0)
    
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def extract_visual_keywords(caption):
    """
    Extract only VISUAL/OBJECT words from a caption.

    The keyword probe asks CLIP 'does this image contain X?' — this only makes
    sense for concrete visual subjects (nouns: dog, car, person, building).

    It breaks for:
      - Opinion words:   fake, real, weird, beautiful, ugly
      - Question words:  why, how, what, does, this, looks
      - Abstract words:  sense, reason, feel, seems
      - Action verbs:    running, laughing (ok for CLIP but not as reliable)
      - Short fillers:   the, is, at, on

    We filter aggressively: only keep words that are likely to be
    concrete visual nouns or places recognizable by CLIP.
    """
    # Full stopword list — opinion + question + abstract + grammar
    non_visual = {
        # grammar/function
        'a','an','the','is','are','was','were','be','been','has','have','had',
        'do','does','did','will','would','can','could','should','may','might',
        'of','in','on','at','to','for','with','by','from','and','or','but',
        'not','this','that','it','its','very','just','some','so','as','if',
        'than','then','there','their','they','we','our','my','your','his','her',
        # opinion / evaluation
        'fake','real','true','false','bad','good','nice','ugly','beautiful',
        'weird','strange','odd','wrong','right','correct','incorrect','valid',
        'looks','look','seem','seems','appear','appears','feel','feels',
        'think','believe','know','see','say','said','tell','told',
        # questions / meta
        'why','how','what','when','where','who','which','whether',
        'does','did','was','is','are','can','could','should','would',
        # abstract
        'sense','reason','way','fact','thing','stuff','much','many','lot',
        'like','about','even','also','only','still','already','again',
        'because','since','though','although','however','therefore',
    }
    words = caption.lower().split()
    # Strip punctuation from each word
    import re
    cleaned = [re.sub(r"[^a-z]", "", w) for w in words]
    # Keep only words that are:
    # - not in non_visual list
    # - at least 3 characters
    # - not a number
    visual = [w for w in cleaned
              if w and len(w) >= 3
              and w not in non_visual
              and not w.isdigit()]
    return visual[:5]  # max 5 keywords


def analyze_caption_consistency(image_pil, caption, cm, cp, clip_ok):
    """
    Measures image-caption semantic consistency using CLIP.

    KEY FIX: Keyword probing now only runs on VISUAL words (nouns, places,
    objects) — not opinion, question, or abstract words. 'fake', 'looks',
    'why', 'this' are not visual concepts CLIP can verify in an image.

    The overall match score uses the full caption semantically (CLIP embeds
    meaning, not just words), so opinion sentences still affect consistency.
    """
    if not clip_ok or not caption.strip():
        return None
    try:
        uc = caption.strip()

        # ── MATCH SCORE: full semantic match ──────────────────────
        # Compare full caption against 8 diverse alternatives.
        # CLIP's text encoder embeds meaning — so "why does this look fake"
        # will score LOW against most real images (good signal).
        alts = [uc,
                "a photograph of something completely unrelated",
                "an abstract texture or pattern",
                "a blank white or black background",
                "a completely different scene",
                "random objects with no coherent subject",
                "an outdoor landscape with no people",
                "an indoor scene with furniture",
                "a close-up of food or drink"]
        i1 = cp(text=alts, images=image_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            p1 = torch.softmax(cm(**i1).logits_per_image.squeeze(0), dim=0)
            s1 = max(0.0, min(1.0, (float(p1[0]) - 0.111) / 0.889))

        hn = [uc, f"nothing to do with: {uc}", f"opposite of: {uc}"]
        i2 = cp(text=hn, images=image_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            p2 = torch.softmax(cm(**i2).logits_per_image.squeeze(0), dim=0)
            s2 = max(0.0, min(1.0, (float(p2[0]) - 0.333) / 0.667))

        match = 0.55*s1 + 0.45*s2

        # ── KEYWORD SCORES: visual nouns only ─────────────────────
        # extract_visual_keywords() strips opinion/question/abstract words
        # so only concrete visual concepts are probed (e.g. 'dog', 'car')
        visual_words = extract_visual_keywords(uc)
        ws = {}

        for word in visual_words:
            pos = [f"a photo of {word}", f"an image showing {word}",
                   f"a picture containing {word}", f"a scene with {word}"]
            neg = [f"a photo with no {word}", f"an image without {word}",
                   f"no {word} in this picture", f"a scene lacking {word}"]
            iw = cp(text=pos+neg, images=image_pil, return_tensors="pt", padding=True)
            with torch.no_grad():
                pw = torch.softmax(cm(**iw).logits_per_image.squeeze(0), dim=0)
            ps_ = float(pw[:4].sum())
            ns_ = float(pw[4:].sum())
            ws[word] = ps_ / (ps_ + ns_ + 1e-8)

        # If no visual keywords found (e.g. pure opinion sentence), note that
        no_visual_keywords = len(visual_words) == 0

        return {
            'match_score'        : match,
            'word_scores'        : ws,
            'caption'            : uc,
            'no_visual_keywords' : no_visual_keywords,
            'visual_words_found' : visual_words,
        }
    except Exception:
        return None

# def combine_scores(sbi, clip_s, dct_ai, ela_ps, dct_ps, clip_ok, cap_analysis=None, ps_det=0.0, ss_det=0.0, meta=0.0, real_conf=0.0, hm_mean=0.0):
#     dct_ai_c = min(dct_ai, 0.80)
#     ela_c    = min(ela_ps, 0.75)
#     dct_ps_c = min(dct_ps, 0.70)

#     is_global_ai = hm_mean > 0.75
#     is_local_splice = 0.02 < hm_mean <= 0.75

#     # ── STEP 1: The Hallucination Veto ──
#     # Run this FIRST. If PyTorch is screaming fake but forensics are totally dead,
#     # crush the PyTorch score immediately so it stops polluting the math.
#     heuristics_dead = (ela_ps < 0.35) and (dct_ps < 0.30)
#     if sbi > 0.70 and heuristics_dead:
#         sbi *= 0.25               
#         is_local_splice = False   

#     # ── STEP 2: Real Photo Suppression (The 0.68 Boundary) ──
#     # If the image has optical camera properties (bokeh, phone ratio) it suppresses
#     # the AI flags. BUT if CLIP is highly confident (> 0.68), assume the AI is just
#     # faking those camera properties and DO NOT suppress.
#     if real_conf > 0.30 and clip_s < 0.68:
#         clip_s = max(0.0, clip_s - (real_conf * 0.60))
#         sbi    = max(0.0, sbi - (real_conf * 0.60)) 
    
#     # ── AI Track Calculation ──
#     if clip_ok:
#         if is_global_ai:
#             ai = 0.60 * clip_s + 0.30 * sbi + 0.10 * dct_ai_c
#         elif is_local_splice and sbi > 0.55:
#             ai = 0.25 * clip_s + 0.55 * sbi + 0.20 * dct_ai_c
#         elif sbi < 0.25:
#             ai = 0.65 * clip_s + 0.15 * sbi + 0.20 * dct_ai_c
#         else:
#             ai = 0.50 * clip_s + 0.30 * sbi + 0.20 * dct_ai_c
#     else:
#         ai = 0.65 * sbi + 0.35 * dct_ai_c

#     # ── Edit/Photoshop track ──
#     ps_det_c = min(ps_det, 0.90)
    
#     # JPEG Artifact Suppression
#     if sbi < 0.25:
#         ela_c *= 0.40
#         dct_ps_c *= 0.40
#         ps_det_c *= 0.50
    
#     if is_global_ai:
#         ps = 0.10 * sbi + 0.40 * ela_c + 0.20 * dct_ps_c + 0.30 * ps_det_c
#     else:
#         ps = 0.45 * sbi + 0.20 * ela_c + 0.15 * dct_ps_c + 0.20 * ps_det_c

#     # ── Caption mismatch boost ──
#     if cap_analysis is not None:
#         ms = cap_analysis['match_score']
#         if ms < 0.30:
#             ai = min(1.0, ai + 0.10)
#         elif ms < 0.45:
#             ai = min(1.0, ai + 0.05)

#     # ── Multi-signal agreement boost ──
#     if clip_ok and clip_s > 0.65 and dct_ai > 0.55:
#         ai = min(1.0, ai + 0.05)
#     if sbi > 0.65 and ela_ps > 0.55:
#         ps = min(1.0, ps + 0.05)

#     # ── Metadata modulation ──
#     if meta > 0.20:
#         ai = min(1.0, ai + meta * 0.35)
#     elif meta < -0.10:
#         ai = max(0.0, ai + meta * 0.40) 

#     # ── High-confidence single-signal override ──
#     if clip_ok and clip_s > 0.80 and sbi < 0.30:
#         ai = max(ai, 0.75)  
#     if sbi > 0.80 and is_local_splice:
#         ps = max(ps, 0.75) 

#     # ── Per-type calibrated final blend ──
#     gap = ai - ps
#     if gap > 0.25:
#         final = ai * 0.75 + ps * 0.25
#     elif gap < -0.15:
#         final = ps * 0.75 + ai * 0.25
#     else:
#         final = max(ai, ps) * 0.60 + min(ai, ps) * 0.40

#     return float(min(1.0, final)), float(ai), float(ps)

# second part that worked
# def combine_scores(sbi, clip_s, dct_ai, ela_ps, dct_ps, clip_ok, cap_analysis=None, ps_det=0.0, ss_det=0.0, meta=0.0, real_conf=0.0, hm_mean=0.0):
    
#     # ── THE L1 QUARANTINE (THE ONCE-AND-FOR-ALL FIX) ──
#     # A true Photoshop splice (copy-paste) is localized. It rarely covers more 
#     # than 30-40% of the image. If L1 (SBI) is screaming that it found a splice, 
#     # but the heatmap is highlighting half the image, it is hallucinating on 
#     # natural camera blur (bokeh). We muzzle it by forcing its score to near-zero.
#     if sbi > 0.50 and hm_mean > 0.40:
#         sbi = 0.05  
        
#     dct_ai_c = min(dct_ai, 0.80)
#     ela_c    = min(ela_ps, 0.75)
#     dct_ps_c = min(dct_ps, 0.70)

#     is_global_ai = hm_mean > 0.75
#     is_local_splice = 0.02 < hm_mean <= 0.40  # Tightened definition of a splice

#     # ── STEP 1: The Hallucination Veto ──
#     heuristics_dead = (ela_ps < 0.35) and (dct_ps < 0.30)
#     if sbi > 0.70 and heuristics_dead:
#         sbi *= 0.25               
#         is_local_splice = False   

#     # ── STEP 2: Real Photo Suppression ──
#     if real_conf > 0.30 and clip_s < 0.68:
#         clip_s = max(0.0, clip_s - (real_conf * 0.60))
#         sbi    = max(0.0, sbi - (real_conf * 0.60)) 
    
#     # ── AI Track Calculation ──
#     if clip_ok:
#         if is_global_ai:
#             ai = 0.60 * clip_s + 0.30 * sbi + 0.10 * dct_ai_c
#         elif is_local_splice and sbi > 0.55:
#             ai = 0.25 * clip_s + 0.55 * sbi + 0.20 * dct_ai_c
#         elif sbi < 0.25:
#             ai = 0.65 * clip_s + 0.15 * sbi + 0.20 * dct_ai_c
#         else:
#             ai = 0.50 * clip_s + 0.30 * sbi + 0.20 * dct_ai_c
#     else:
#         ai = 0.65 * sbi + 0.35 * dct_ai_c

#     # ── Edit/Photoshop Track ──
#     ps_det_c = min(ps_det, 0.90)
    
#     # JPEG Artifact Suppression
#     if sbi < 0.25:
#         ela_c *= 0.40
#         dct_ps_c *= 0.40
#         ps_det_c *= 0.50
    
#     if is_global_ai:
#         ps = 0.10 * sbi + 0.40 * ela_c + 0.20 * dct_ps_c + 0.30 * ps_det_c
#     else:
#         ps = 0.45 * sbi + 0.20 * ela_c + 0.15 * dct_ps_c + 0.20 * ps_det_c

#     # ── Caption Mismatch Boost ──
#     if cap_analysis is not None:
#         ms = cap_analysis['match_score']
#         if ms < 0.30:
#             ai = min(1.0, ai + 0.10)
#         elif ms < 0.45:
#             ai = min(1.0, ai + 0.05)

#     # ── Multi-Signal Agreement Boost ──
#     if clip_ok and clip_s > 0.65 and dct_ai > 0.55:
#         ai = min(1.0, ai + 0.05)
#     if sbi > 0.65 and ela_ps > 0.55:
#         ps = min(1.0, ps + 0.05)

#     # ── Metadata Modulation ──
#     if meta > 0.20:
#         ai = min(1.0, ai + meta * 0.35)
#     elif meta < -0.10:
#         ai = max(0.0, ai + meta * 0.40) 

#     # ── High-Confidence Single-Signal Override ──
#     if clip_ok and clip_s > 0.80 and sbi < 0.30:
#         ai = max(ai, 0.75)  
#     if sbi > 0.80 and is_local_splice:
#         ps = max(ps, 0.75) 

#     # ── Per-Type Calibrated Final Blend ──
#     gap = ai - ps
#     if gap > 0.25:
#         final = ai * 0.75 + ps * 0.25
#     elif gap < -0.15:
#         final = ps * 0.75 + ai * 0.25
#     else:
#         final = max(ai, ps) * 0.60 + min(ai, ps) * 0.40

#     return float(min(1.0, final)), float(ai), float(ps)


# main def that worked lastone
# def combine_scores(sbi, clip_s, dct_ai, ela_ps, dct_ps, clip_ok, cap_analysis=None, ps_det=0.0, ss_det=0.0, meta=0.0, real_conf=0.0, hm_mean=0.0):
    
#     # ── THE L1 QUARANTINE ──
#     if sbi > 0.50 and hm_mean > 0.40:
#         sbi = 0.05  
        
#     dct_ai_c = min(dct_ai, 0.80)
#     ela_c    = min(ela_ps, 0.75)
#     dct_ps_c = min(dct_ps, 0.70)

#     is_global_ai = hm_mean > 0.75
#     is_local_splice = 0.02 < hm_mean <= 0.40

#     # ── STEP 1: The Hallucination Veto (FIXED) ──
#     heuristics_dead = (ela_ps < 0.35) and (dct_ps < 0.30)
#     # ONLY veto if the U-Net heatmap DOES NOT confirm a tight, localized splice
#     if sbi > 0.70 and heuristics_dead and not is_local_splice:
#         sbi *= 0.25               
#         is_local_splice = False   

#     # ── STEP 2: Real Photo Suppression ──
#     if real_conf > 0.30 and clip_s < 0.68:
#         clip_s = max(0.0, clip_s - (real_conf * 0.60))
#         sbi    = max(0.0, sbi - (real_conf * 0.60)) 
    
#     # ── AI Track Calculation (FIXED ROUTING) ──
#     if clip_ok:
#         if is_global_ai:
#             ai = 0.60 * clip_s + 0.30 * sbi + 0.10 * dct_ai_c
#         elif is_local_splice:
#             # STOP the PyTorch model from leaking into the AI score
#             ai = 0.70 * clip_s + 0.05 * sbi + 0.25 * dct_ai_c
#         elif sbi < 0.25:
#             ai = 0.65 * clip_s + 0.15 * sbi + 0.20 * dct_ai_c
#         else:
#             ai = 0.50 * clip_s + 0.30 * sbi + 0.20 * dct_ai_c
#     else:
#         ai = 0.65 * sbi + 0.35 * dct_ai_c

#     # ── Edit/Photoshop Track (FIXED ROUTING) ──
#     ps_det_c = min(ps_det, 0.90)
    
#     if sbi < 0.25:
#         ela_c *= 0.40
#         dct_ps_c *= 0.40
#         ps_det_c *= 0.50
    
#     if is_global_ai:
#         ps = 0.10 * sbi + 0.40 * ela_c + 0.20 * dct_ps_c + 0.30 * ps_det_c
#     elif is_local_splice:
#         # FORCE the heavy PyTorch score into the Photoshop bucket
#         ps = 0.65 * sbi + 0.15 * ela_c + 0.10 * dct_ps_c + 0.10 * ps_det_c
#     else:
#         ps = 0.45 * sbi + 0.20 * ela_c + 0.15 * dct_ps_c + 0.20 * ps_det_c

#     # ── Caption Mismatch Boost ──
#     if cap_analysis is not None:
#         ms = cap_analysis['match_score']
#         if ms < 0.30:
#             ai = min(1.0, ai + 0.10)
#         elif ms < 0.45:
#             ai = min(1.0, ai + 0.05)

#     # ── Multi-Signal Agreement Boost ──
#     if clip_ok and clip_s > 0.65 and dct_ai > 0.55:
#         ai = min(1.0, ai + 0.05)
#     if sbi > 0.65 and ela_ps > 0.55:
#         ps = min(1.0, ps + 0.05)

#     # ── Metadata Modulation ──
#     if meta > 0.20:
#         ai = min(1.0, ai + meta * 0.35)
#     elif meta < -0.10:
#         ai = max(0.0, ai + meta * 0.40) 

#     # ── High-Confidence Single-Signal Override ──
#     if clip_ok and clip_s > 0.80 and sbi < 0.30:
#         ai = max(ai, 0.75)  
#     if sbi > 0.80 and is_local_splice:
#         ps = max(ps, 0.75) 

#     # ── Per-Type Calibrated Final Blend ──
#     gap = ai - ps
#     if gap > 0.25:
#         final = ai * 0.75 + ps * 0.25
#     elif gap < -0.15:
#         final = ps * 0.75 + ai * 0.25
#     else:
#         final = max(ai, ps) * 0.60 + min(ai, ps) * 0.40

#     return float(min(1.0, final)), float(ai), float(ps)

# worked for some but not for each
# def combine_scores(sbi, clip_s, dct_ai, ela_ps, dct_ps, clip_ok, cap_analysis=None, ps_det=0.0, ss_det=0.0, meta=0.0, real_conf=0.0, hm_mean=0.0):
    
#     # ── THE L1 QUARANTINE ──
#     # Broadened slightly to 0.50 to ensure larger splices don't get quarantined
#     if sbi > 0.50 and hm_mean > 0.60:
#         sbi = 0.05  
        
#     dct_ai_c = min(dct_ai, 0.80)
#     ela_c    = min(ela_ps, 0.75)
#     dct_ps_c = min(dct_ps, 0.70)

#     is_global_ai = hm_mean > 0.75
#     # Broadened heatmap tolerance to ensure TIFF splices are recognized
#     is_local_splice = 0.005 < hm_mean <= 0.60

#     # ── STEP 1: The Hallucination Veto (TIFF FIX) ──
#     heuristics_dead = (ela_ps < 0.35) and (dct_ps < 0.30)
#     # If SBI is absolute (>0.90), trust it. TIFFs naturally have dead heuristics.
#     if sbi > 0.70 and sbi < 0.85 and heuristics_dead and not is_local_splice:
#         sbi *= 0.25               
#         is_local_splice = False   

#     # ── STEP 2: Real Photo Suppression (SPLICING FIX) ──
#     if real_conf > 0.30 and clip_s < 0.68:
#         clip_s = max(0.0, clip_s - (real_conf * 0.60))
#         # DO NOT suppress the PyTorch model if it is highly confident in a splice!
#         if sbi < 0.85:
#             sbi = max(0.0, sbi - (real_conf * 0.60)) 
    
#     # ── AI Track Calculation ──
#     if clip_ok:
#         if is_global_ai:
#             ai = 0.60 * clip_s + 0.30 * sbi + 0.10 * dct_ai_c
#         elif is_local_splice:
#             ai = 0.70 * clip_s + 0.05 * sbi + 0.25 * dct_ai_c
#         elif sbi < 0.25:
#             ai = 0.65 * clip_s + 0.15 * sbi + 0.20 * dct_ai_c
#         else:
#             ai = 0.50 * clip_s + 0.30 * sbi + 0.20 * dct_ai_c
#     else:
#         ai = 0.65 * sbi + 0.35 * dct_ai_c

#     # ── Edit/Photoshop Track ──
#     ps_det_c = min(ps_det, 0.90)
    
#     if sbi < 0.25:
#         ela_c *= 0.40
#         dct_ps_c *= 0.40
#         ps_det_c *= 0.50
    
#     if is_global_ai:
#         ps = 0.10 * sbi + 0.40 * ela_c + 0.20 * dct_ps_c + 0.30 * ps_det_c
#     elif is_local_splice:
#         ps = 0.65 * sbi + 0.15 * ela_c + 0.10 * dct_ps_c + 0.10 * ps_det_c
#     else:
#         ps = 0.45 * sbi + 0.20 * ela_c + 0.15 * dct_ps_c + 0.20 * ps_det_c

#     # ── Caption Mismatch Boost ──
#     if cap_analysis is not None:
#         ms = cap_analysis['match_score']
#         if ms < 0.30:
#             ai = min(1.0, ai + 0.10)
#         elif ms < 0.45:
#             ai = min(1.0, ai + 0.05)

#     # ── Multi-Signal Agreement Boost ──
#     if clip_ok and clip_s > 0.65 and dct_ai > 0.55:
#         ai = min(1.0, ai + 0.05)
#     if sbi > 0.65 and ela_ps > 0.55:
#         ps = min(1.0, ps + 0.05)

#     # ── Metadata Modulation ──
#     if meta > 0.20:
#         ai = min(1.0, ai + meta * 0.35)
#     elif meta < -0.10:
#         ai = max(0.0, ai + meta * 0.40) 

#     # ── High-Confidence Single-Signal Override ──
#     if clip_ok and clip_s > 0.80 and sbi < 0.30:
#         ai = max(ai, 0.75)  
#     if sbi > 0.80 and is_local_splice:
#         ps = max(ps, 0.75) 

#     # ── Per-Type Calibrated Final Blend ──
#     gap = ai - ps
#     if gap > 0.25:
#         final = ai * 0.75 + ps * 0.25
#     elif gap < -0.15:
#         final = ps * 0.75 + ai * 0.25
#     else:
#         final = max(ai, ps) * 0.60 + min(ai, ps) * 0.40

#     return float(min(1.0, final)), float(ai), float(ps)

def combine_scores(sbi, clip_s, dct_ai, ela_ps, dct_ps, clip_ok, cap_analysis=None, ps_det=0.0, ss_det=0.0, meta=0.0, real_conf=0.0, hm_mean=0.0):

    if sbi > 0.50 and hm_mean > 0.60:
        sbi = 0.05  
        
    dct_ai_c = min(dct_ai, 0.80)
    ela_c    = min(ela_ps, 0.75)
    dct_ps_c = min(dct_ps, 0.70)

    is_global_ai = hm_mean > 0.75

    is_local_splice = 0.005 < hm_mean <= 0.60

    # ── STEP 1: The Hallucination Veto (TIFF ARMOR) ──
    heuristics_dead = (ela_ps < 0.35) and (dct_ps < 0.30)
    # If SBI is > 0.85, it bypasses the veto entirely, protecting it from 
    # being crushed by dead TIFF heuristics, regardless of how bad the heatmap looks.
    if sbi > 0.70 and sbi < 0.85 and heuristics_dead and not is_local_splice:
        sbi *= 0.25               
        is_local_splice = False   

    # ── STEP 2: Real Photo Suppression ──
    if real_conf > 0.30 and clip_s < 0.68:
        clip_s = max(0.0, clip_s - (real_conf * 0.60))
        # Protect highly confident splice detections from being suppressed
        if sbi < 0.85:
            sbi = max(0.0, sbi - (real_conf * 0.60)) 
    
    # ── AI Track Calculation ──
    if clip_ok:
        if is_global_ai:
            ai = 0.60 * clip_s + 0.30 * sbi + 0.10 * dct_ai_c
        elif is_local_splice:
            ai = 0.70 * clip_s + 0.05 * sbi + 0.25 * dct_ai_c
        elif sbi < 0.25:
            ai = 0.65 * clip_s + 0.15 * sbi + 0.20 * dct_ai_c
        else:
            ai = 0.50 * clip_s + 0.30 * sbi + 0.20 * dct_ai_c
    else:
        ai = 0.65 * sbi + 0.35 * dct_ai_c

    # ── Edit/Photoshop Track ──
    ps_det_c = min(ps_det, 0.90)
    
    if sbi < 0.25:
        ela_c *= 0.40
        dct_ps_c *= 0.40
        ps_det_c *= 0.50
    
    if is_global_ai:
        ps = 0.10 * sbi + 0.40 * ela_c + 0.20 * dct_ps_c + 0.30 * ps_det_c
    elif is_local_splice:
        ps = 0.65 * sbi + 0.15 * ela_c + 0.10 * dct_ps_c + 0.10 * ps_det_c
    else:
        ps = 0.45 * sbi + 0.20 * ela_c + 0.15 * dct_ps_c + 0.20 * ps_det_c

    # ── Multi-Signal Agreement Boost ──
    if clip_ok and clip_s > 0.65 and dct_ai > 0.55:
        ai = min(1.0, ai + 0.05)
    if sbi > 0.65 and ela_ps > 0.55:
        ps = min(1.0, ps + 0.05)

    # ── Metadata Modulation ──
    if meta > 0.20:
        ai = min(1.0, ai + meta * 0.35)
    elif meta < -0.10:
        ai = max(0.0, ai + meta * 0.40) 

    # ── High-Confidence Single-Signal Override ──
    if clip_ok and clip_s > 0.80 and sbi < 0.30:
        ai = max(ai, 0.75)  
    if sbi > 0.80 and is_local_splice:
        ps = max(ps, 0.75) 

    # ── Per-Type Calibrated Final Blend ──
    gap = ai - ps
    if gap > 0.25:
        final = ai * 0.75 + ps * 0.25
    elif gap < -0.15:
        final = ps * 0.75 + ai * 0.25
    else:
        final = max(ai, ps) * 0.60 + min(ai, ps) * 0.40

    return float(min(1.0, final)), float(ai), float(ps)

def get_verdict(final, ai, ps, threshold, ss=0.0, ps_det=0.0):
    # ── Screenshot — runs before everything else ──
    if ss > 0.25:
        return "SCREENSHOT", f"Computer/UI-generated image ({ss*100:.0f}% confidence)", "🖥", "vc-amber", "verdict-manip"

    # ── THE FIX: Trust the User's Threshold Slider! ──
    # We removed the min(threshold, 0.28) hardcap. 
    ai_threshold = threshold
    ps_threshold = threshold

    if ai > ps:
        if final < ai_threshold:
            return "AUTHENTIC", "No AI generation detected", "✓", "vc-green", "verdict-authentic"
    else:
        if final < ps_threshold:
            return "AUTHENTIC", "No manipulation detected", "✓", "vc-green", "verdict-authentic"

    # ── Photoshop — high ps_det with ps track dominant ──
    if ps_det > 0.55 and ps > ai:
        return "PHOTOSHOPPED", "Copy-move or compositing edits detected", "✗", "vc-red", "verdict-photoshop"

    # ── AI verdict ──
    if ai > ps:
        if ai > 0.72:
            return "AI GENERATED",  "Strong AI/GAN/diffusion signal",   "⚠", "vc-purple", "verdict-ai"
        if ai > 0.48:
            return "LIKELY AI",     "Probable AI-generated image",       "◐", "vc-purple", "verdict-ai"

    # ── Photoshop/splice verdict ──
    if ps > ai:
        if ps > 0.68:
            return "PHOTOSHOPPED",  "Splice/Photoshop manipulation detected", "✗", "vc-red",   "verdict-photoshop"
        return     "MANIPULATED",   "Image shows signs of editing",           "⚠", "vc-amber", "verdict-manip"

    return "MANIPULATED", "Manipulation detected", "⚠", "vc-amber", "verdict-manip"

def lstyle(s, kind='normal'):
    if kind == 'purple':
        if s > 0.55: return 'layer-row-purple', '#a855f7', 'HIGH'
        if s > 0.35: return 'layer-row-amber',  '#f59e0b', 'MED'
        return              'layer-row-green',  '#22c55e', 'LOW'
    if s > 0.65: return 'layer-row-red',   '#ef4444', 'HIGH'
    if s > 0.45: return 'layer-row-amber', '#f59e0b', 'MED'
    return              'layer-row-green', '#22c55e', 'LOW'


# =============================================================
# LOAD MODELS
# =============================================================
with st.spinner("Loading models..."):
    sbi_model, tokenizer, device        = load_sbi_model()
    clip_model, clip_processor, clip_ok = load_clip_model()


# =============================================================
# CONTROLS (moved inline — sidebar toggle was unreliable)
# =============================================================
is_light = (st.session_state.get('theme', 'dark') == 'light')


threshold = 0.30
region_thr = 0.45

# 2. Put the Header and the Theme Button in the same horizontal row
header_col, btn_col = st.columns([8.5, 1.5], vertical_alignment="center")

with header_col:
    # We removed the bottom border from the CSS class here so it blends perfectly
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; padding:0.5rem 0;">
        <div class="s-logo-row">
            <div class="s-logo">🛡️</div>
            <div>
                <div class="s-title">SENTINEL</div>
                <div class="s-subtitle">MEDIA FORGERY DETECTION</div>
            </div>
        </div>
        <div class="s-badge">{'● ALL SYSTEMS ONLINE' if clip_ok else '● CLIP OFFLINE'}</div>
    </div>""", unsafe_allow_html=True)

with btn_col:
    btn_label = "🌙 Dark" if is_light else "☀️ Light"
    if st.button(btn_label, use_container_width=True, type="secondary"):
        st.session_state.theme      = 'light' if not is_light else 'dark'
        st.session_state.light_mode = not is_light
        st.rerun()

st.markdown("---")

# =============================================================
# MAIN TWO-COLUMN LAYOUT
# =============================================================
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown('<div class="sec-label">Evidence Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Image", type=["jpg","jpeg","png","webp", 'tiff', 'tif'], label_visibility="collapsed")
    st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);margin-top:-0.3rem;margin-bottom:0.75rem">JPG · PNG · WEBP</div>', unsafe_allow_html=True)
    caption  = st.text_area("Caption", placeholder="Describe the image (optional)...", height=75, label_visibility="collapsed")
    st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);margin-top:-0.3rem;margin-bottom:0.8rem">Optional — enables caption + AI caption analysis</div>', unsafe_allow_html=True)
    run_btn  = st.button("▶  Run Analysis", use_container_width=True, disabled=(uploaded is None))

    if uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Preview</div>', unsafe_allow_html=True)
        st.image(uploaded, use_container_width=True)
        img_check = Image.open(uploaded)
        fbytes    = uploaded.getvalue()
        st.markdown(f'<div class="info-row">File: <span class="iv">{uploaded.name}</span> &nbsp;·&nbsp; <span class="iv">{img_check.size[0]}×{img_check.size[1]}px</span> &nbsp;·&nbsp; <span class="iv">{len(fbytes)/1024:.1f} KB</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state" style="margin-top:0.5rem"><div class="empty-icon">📁</div><div class="empty-text">Upload an image to begin</div></div>', unsafe_allow_html=True)


with col_r:
    st.markdown('<div class="sec-label">Analysis Report</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        image_pil = Image.open(uploaded)

        # with st.spinner("L1 — SBI splice detection..."):
        #     sbi_score, heatmap_np, image_np = run_sbi_inference(image_pil, caption, sbi_model, tokenizer, device)
        with st.spinner("L1 — SBI splice detection..."):
            # Replace 'caption' with a blank string or a generic, hardcoded prompt
            # so the user's text can never influence the pixel math.
            safe_sbi_prompt = ""  # (If your specific model requires text, change this to "image" or "manipulated region")
            sbi_score, heatmap_np, image_np = run_sbi_inference(image_pil, safe_sbi_prompt, sbi_model, tokenizer, device)
        with st.spinner("L2 — Visual AI detection..."):
            visual_score  = visual_ai_detection(image_pil)
            real_conf     = real_photo_confidence(image_pil)
        with st.spinner("L2b — CLIP semantic detection..."):
            clip_raw, clip_fired = (clip_ai_detection(image_pil, clip_model, clip_processor) if clip_ok else (0.5, False))
            # Combine: visual heuristics are reliable, CLIP adds semantic signal
            if clip_fired and clip_raw != 0.5:
                if visual_score > 0.70:
                    clip_score = 0.50 * clip_raw + 0.50 * visual_score
                elif visual_score > 0.50:
                    clip_score = 0.60 * clip_raw + 0.40 * visual_score
                else:
                    clip_score = 0.70 * clip_raw + 0.30 * visual_score
            else:
                # CLIP failed — use visual heuristics directly as clip_score
                clip_score = visual_score
        with st.spinner("L3 — DCT frequency..."):
            dct_ai, dct_ps, dct_d = dct_frequency_analysis(image_pil)
        with st.spinner("L4 — ELA Photoshop..."):
            ela_score, ela_gray, ela_vis = ela_analysis(image_pil)
        with st.spinner("L5 — Screenshot detection..."):
            ss_score, ss_details = screenshot_detection(image_pil)
        with st.spinner("L6 — Photoshop/copy-move detection..."):
            ps_det_score, ps_det_details = photoshop_detection(image_pil)
        with st.spinner("L7 — Metadata analysis..."):
            meta_score, meta_details = metadata_analysis(image_pil)
        with st.spinner("L8 — Caption analysis..."):
            cap_cons   = analyze_caption_consistency(image_pil, caption, clip_model, clip_processor, clip_ok)
            cap_ai_res = detect_ai_caption(caption, clip_model, clip_processor, clip_ok)
        with st.spinner("Computing verdict..."):
            hm_mean = float(heatmap_np.mean()) 
            final, ai_score, ps_score = combine_scores(sbi_score, clip_score, dct_ai, ela_score, dct_ps, clip_ok, cap_cons, ps_det_score, ss_score, meta_score, real_conf, hm_mean)
            v_lbl, v_desc, v_icon, v_col, v_cls = get_verdict(final, ai_score, ps_score, threshold, ss_score, ps_det_score)

        # Verdict
        st.markdown(f"""
        <div class="verdict-wrap {v_cls}">
            <div class="verdict-icon">{v_icon}</div>
            <div class="verdict-label {v_col}">{v_lbl}</div>
            <div class="verdict-desc">{v_desc} · Confidence {final*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # Score pills — 4 pills including raw SBI model output
        fc   = '#ef4444' if v_lbl != 'AUTHENTIC' else '#22c55e'
        sbi_col = '#ef4444' if sbi_score > threshold else '#22c55e'
        st.markdown(f"""
        <div class="pill-row" style="grid-template-columns:repeat(4,1fr)">
            <div class="pill">
                <div class="pill-val" style="color:{fc}">{final*100:.1f}%</div>
                <div class="pill-lbl">Final Score</div>
            </div>
            <div class="pill">
                <div class="pill-val" style="color:{sbi_col}">{sbi_score*100:.1f}%</div>
                <div class="pill-lbl">Model (SBI)</div>
            </div>
            <div class="pill">
                <div class="pill-val" style="color:#a855f7">{ai_score*100:.1f}%</div>
                <div class="pill-lbl">AI Score</div>
            </div>
            <div class="pill">
                <div class="pill-val" style="color:#f59e0b">{ps_score*100:.1f}%</div>
                <div class="pill-lbl">Edit Score</div>
            </div>
        </div><br>""", unsafe_allow_html=True)

        # FIX 8 — Raw model output debug panel (always visible, not hidden)
        with st.expander("🔬 Raw Signal Breakdown", expanded=False):
            st.markdown(f"""
            | Signal | Raw Value | Interpretation |
            |--------|-----------|----------------|
            | **SBI Model logit → prob** | `{sbi_score:.4f}` | {'🔴 Fake' if sbi_score > 0.5 else '🟢 Real'} (threshold 0.5) |
            | **CLIP AI detection** | `{clip_score:.4f}` | {'🔴 AI-generated' if clip_score > 0.5 else '🟢 Real photo'} |
            | **DCT AI smoothness** | `{dct_ai:.4f}` | {'🔴 AI-smooth' if dct_ai > 0.5 else '🟢 Normal freq'} |
            | **DCT JPEG block** | `{dct_ps:.4f}` | {'🟡 Block artifacts' if dct_ps > 0.4 else '🟢 Normal'} |
            | **ELA edit score** | `{ela_score:.4f}` | {'🔴 Edits detected' if ela_score > 0.5 else '🟢 Uniform compression'} |
            | **Combined AI score** | `{ai_score:.4f}` | Weighted: 0.50·SBI + 0.35·CLIP + 0.15·DCT |
            | **Combined Edit score** | `{ps_score:.4f}` | Weighted: 0.60·SBI + 0.25·ELA + 0.15·DCT |
            | **Final score** | `{final:.4f}` | 0.55·AI + 0.45·Edit |
            | **Threshold** | `{threshold:.2f}` | Adjustable in sidebar |
            """)

        # Layers
        st.markdown('<div class="sec-label">Detection Layers</div>', unsafe_allow_html=True)
        for icon, name, desc, score, active, kind in [
            ("🔬","L1 — SBI Splice",       "Copy-paste & region splicing",    sbi_score,  True,     'normal'),
            ("🧠","L2 — CLIP AI",          "GAN · Diffusion · MidJourney",    clip_score, clip_ok,  'purple'),
            ("📡","L3 — DCT (AI artifacts)","Frequency smoothness pattern",    dct_ai,     True,     'purple'),
            ("🖼️","L3b — DCT (JPEG edits)", "8×8 block artifact detection",    dct_ps,     True,     'normal'),
            ("🔍","L4 — ELA Photoshop",    "Error Level Analysis — edits",    ela_score,  True,     'normal'),
        ]:
            if active:
                rc, sc, rl = lstyle(score, kind)
                st.markdown(f"""
                <div class="layer-row {rc}">
                    <div class="layer-icon">{icon}</div>
                    <div class="layer-info">
                        <div class="layer-name">{name}</div>
                        <div class="layer-desc">{desc}</div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{score*100:.0f}%;background:{sc}"></div></div>
                    </div>
                    <div class="layer-right">
                        <div class="layer-score" style="color:{sc}">{score*100:.1f}%</div>
                        <div class="layer-risk" style="color:{sc}">{rl}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="layer-row layer-row-dim"><div class="layer-icon">{icon}</div><div class="layer-info"><div class="layer-name">{name}</div><div class="layer-desc" style="color:var(--red)">Offline</div></div><div class="layer-right"><div class="layer-score" style="color:var(--text-dim)">N/A</div></div></div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="empty-state"><div class="empty-icon">🛡️</div><div class="empty-text">Upload an image and click<br>Run Analysis to begin</div></div>', unsafe_allow_html=True)


# =============================================================
# BOTTOM PANELS
# =============================================================
if uploaded and run_btn:
    st.markdown("<br>", unsafe_allow_html=True)

    # ── REGION HIGHLIGHTING ────────────────────────────────────
    st.markdown('<div class="sec-label">Fake Region Localization</div>', unsafe_allow_html=True)
    hl_img, fake_pct, real_pct, n_regions = segment_regions(heatmap_np, image_np, region_thr)

    rc1, rc2 = st.columns([1, 1], gap="large")
    with rc1:
        t1, t2, t3, t4 = st.tabs(["HIGHLIGHTED", "HEATMAP", "ELA MAP", "ORIGINAL"])
        with t1:
            st.image(hl_img, use_container_width=True)
            st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);text-align:center;margin-top:0.3rem">Color = suspect · Dimmed gray = likely real</div>', unsafe_allow_html=True)
        with t2:
            st.image(create_overlay(image_np, heatmap_np), use_container_width=True)
            st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);text-align:center;margin-top:0.3rem">Red/yellow = high manipulation probability</div>', unsafe_allow_html=True)
        with t3:
            st.image(ela_vis, use_container_width=True)
            st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);text-align:center;margin-top:0.3rem">ELA: Bright = edited region · Dark = original</div>', unsafe_allow_html=True)
        with t4:
            st.image(image_np, use_container_width=True)

    with rc2:
        st.markdown('<div class="sec-label">Region Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rbox rbox-real">
            <div style="font-family:var(--mono);font-size:0.6rem;color:var(--green);letter-spacing:0.08em;margin-bottom:0.25rem">✓ REAL REGION</div>
            <div style="font-family:var(--mono);font-size:1.5rem;font-weight:600;color:var(--green)">{real_pct:.1f}%</div>
            <div style="font-family:var(--mono);font-size:0.6rem;color:var(--text-dim);margin-top:0.15rem">of image appears authentic</div>
        </div>""", unsafe_allow_html=True)

        fc2 = 'var(--red)' if fake_pct > 20 else 'var(--amber)' if fake_pct > 5 else 'var(--green)'
        st.markdown(f"""
        <div class="rbox rbox-fake">
            <div style="font-family:var(--mono);font-size:0.6rem;color:{fc2};letter-spacing:0.08em;margin-bottom:0.25rem">✗ SUSPECT REGION</div>
            <div style="font-family:var(--mono);font-size:1.5rem;font-weight:600;color:{fc2}">{fake_pct:.1f}%</div>
            <div style="font-family:var(--mono);font-size:0.6rem;color:var(--text-dim);margin-top:0.15rem">{n_regions} distinct region{'s' if n_regions!=1 else ''} flagged</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rbox rbox-ai">
            <div style="font-family:var(--mono);font-size:0.6rem;color:var(--purple);letter-spacing:0.08em;margin-bottom:0.5rem">MANIPULATION TYPE</div>
            <div style="font-family:var(--mono);font-size:0.68rem;color:var(--text-dim);line-height:2.2">
                AI Generation: <span style="color:var(--purple);font-weight:600">{ai_score*100:.1f}%</span><br>
                Photoshop / Edit: <span style="color:var(--amber);font-weight:600">{ps_score*100:.1f}%</span><br>
                Dominant: <span style="color:var(--text);font-weight:600">{'AI Generation' if ai_score >= ps_score else 'Manual Edit'}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-row">
            Verdict: <span class="iv">{v_lbl}</span> &nbsp;·&nbsp; Score: <span class="iv">{final*100:.2f}%</span><br>
            Suspect area: <span class="iv">{fake_pct:.1f}%</span> &nbsp;·&nbsp; Regions: <span class="iv">{n_regions}</span><br>
            ELA: <span class="iv">{ela_score*100:.1f}%</span> &nbsp;·&nbsp; DCT block: <span class="iv">{dct_d['block_ratio']:.3f}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CAPTION ANALYSIS ──────────────────────────────────────
    if caption.strip():
        if cap_cons:
            ms = cap_cons.get('match_score', 1.0)
            if ms < 0.30:
                st.error("🚨 **MISATTRIBUTION DETECTED:** The visual contents of this image do not match the caption.")
            elif ms < 0.50:
                st.warning("⚠️ **POOR CAPTION MATCH:** The caption only loosely matches the image contents. Proceed with caution.")

        if cap_ai_res and cap_ai_res.get('label') == 'AI-GENERATED':
            st.info("🤖 **BOT ACTIVITY RISK:** Linguistic analysis indicates the caption was likely generated by an AI language model.")
        
        st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Caption Analysis</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns([1, 1], gap="large")

    with cc1:
        st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);letter-spacing:0.1em;margin-bottom:0.6rem">IMAGE — CAPTION CONSISTENCY</div>', unsafe_allow_html=True)
        if not caption.strip():
            st.markdown('<div class="card-sm" style="text-align:center;padding:2rem"><div style="font-family:var(--mono);font-size:0.68rem;color:var(--text-dim)">No caption provided.</div></div>', unsafe_allow_html=True)
        elif not clip_ok:
            st.markdown('<div class="card-sm"><div style="font-family:var(--mono);font-size:0.68rem;color:var(--text-dim)">CLIP offline.</div></div>', unsafe_allow_html=True)
        elif cap_cons:
            ms = cap_cons['match_score']
            if ms > 0.65:   cc, cl, cls = 'var(--green)',  '✓ CONSISTENT',    'rbox-real'
            elif ms > 0.45: cc, cl, cls = 'var(--amber)',  '◐ PARTIAL MATCH', 'rbox-neutral'
            else:           cc, cl, cls = 'var(--red)',    '✗ INCONSISTENT',  'rbox-fake'

            st.markdown(f"""
            <div class="rbox {cls}">
                <div style="font-family:var(--mono);font-size:0.62rem;color:{cc};letter-spacing:0.06em;margin-bottom:0.3rem">{cl}</div>
                <div style="font-family:var(--mono);font-size:1.5rem;font-weight:600;color:{cc}">{ms*100:.1f}%</div>
                <div style="font-family:var(--mono);font-size:0.6rem;color:var(--text-dim)">caption match score</div>
                <div class="bar-bg" style="margin-top:0.6rem"><div class="bar-fill" style="width:{ms*100:.0f}%;background:{cc}"></div></div>
            </div>""", unsafe_allow_html=True)

            # ── Keyword presence (visual words only) ─────────────
            ws  = cap_cons.get('word_scores', {})
            no_vis = cap_cons.get('no_visual_keywords', False)

            st.markdown('<div style="margin-top:0.6rem;font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);letter-spacing:0.1em;margin-bottom:0.4rem">VISUAL KEYWORD PRESENCE</div>', unsafe_allow_html=True)

            if no_vis or not ws:
                # Caption has no visual/object words — only opinion or question words
                st.markdown(f'''
                <div class="card-sm" style="border-left:3px solid var(--amber);padding:0.8rem 1rem">
                    <div style="font-family:var(--mono);font-size:0.68rem;color:var(--amber);margin-bottom:0.3rem">⚠ NO VISUAL KEYWORDS</div>
                    <div style="font-family:var(--sans);font-size:0.78rem;color:var(--text-dim);line-height:1.7">
                        Caption contains no concrete visual subjects (objects, people, places).
                        Words like <b>"fake"</b>, <b>"looks"</b>, <b>"why"</b> are opinions or questions —
                        not things visible in an image, so keyword probing is skipped.
                        The match score above uses full semantic meaning instead.
                    </div>
                </div>''', unsafe_allow_html=True)
            else:
                for word, score in sorted(ws.items(), key=lambda x: x[1], reverse=True):
                    if score > 0.60:   wc, wl = '#22c55e', 'FOUND'
                    elif score > 0.42: wc, wl = '#f59e0b', 'UNCERTAIN'
                    else:              wc, wl = '#ef4444', 'NOT FOUND'
                    st.markdown(
                        f'<div class="card-sm" style="margin-bottom:0.35rem;padding:0.6rem 0.85rem">' +
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">' +
                        f'<span style="font-family:var(--mono);font-size:0.72rem;color:var(--text)">"{word}"</span>' +
                        f'<span style="font-family:var(--mono);font-size:0.6rem;color:{wc}">{score*100:.0f}% · {wl}</span>' +
                        f'</div><div class="bar-bg"><div class="bar-fill" style="width:{score*100:.0f}%;background:{wc}"></div></div></div>',
                        unsafe_allow_html=True)

                mm_w = [w for w, s in ws.items() if s < 0.42]
                mt_w = [w for w, s in ws.items() if s > 0.60]
                parts = []
                if mt_w: parts.append(f'<span style="color:#22c55e"><b>{", ".join(mt_w)}</b></span> detected in image.')
                if mm_w: parts.append(f'<span style="color:#ef4444"><b>{", ".join(mm_w)}</b></span> NOT found.')
                if not parts: parts.append('No strong keyword signals.')
                st.markdown(f'<div class="card-sm" style="border-left:3px solid {cc};margin-top:0.4rem"><div style="font-family:var(--sans);font-size:0.78rem;color:var(--text);line-height:1.7">{" ".join(parts)}</div></div>', unsafe_allow_html=True)

    with cc2:
        st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);letter-spacing:0.1em;margin-bottom:0.6rem">CAPTION — AI DETECTION</div>', unsafe_allow_html=True)
        if not caption.strip():
            st.markdown('<div class="card-sm" style="text-align:center;padding:2rem"><div style="font-family:var(--mono);font-size:0.68rem;color:var(--text-dim)">No caption provided.</div></div>', unsafe_allow_html=True)
        elif cap_ai_res:
            r   = cap_ai_res
            iai = r['label'] == 'AI-GENERATED'
            ac  = '#a855f7' if iai else '#22c55e'
            st.markdown(f"""
            <div class="rbox {'rbox-ai' if iai else 'rbox-real'}">
                <div style="font-size:1.6rem">{'🤖' if iai else '👤'}</div>
                <div style="font-family:var(--mono);font-size:0.65rem;font-weight:600;color:{ac};letter-spacing:0.08em;margin-top:0.3rem">{r['label']}</div>
                <div style="font-family:var(--mono);font-size:1.5rem;font-weight:600;color:{ac};margin-top:0.2rem">{r['confidence']*100:.1f}%</div>
                <div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim)">confidence</div>
                <div class="bar-bg" style="margin-top:0.6rem"><div class="bar-fill" style="width:{r['score']*100:.0f}%;background:{ac}"></div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div style="margin-top:0.6rem"></div>', unsafe_allow_html=True)
            for sn, sv, sc in [("AI phrase patterns", r['heuristic'], '#a855f7'),
                                 ("Linguistic signal",  r['clip_signal'], '#3b82f6')]:
                st.markdown(f"""
                <div class="card-sm" style="margin-bottom:0.35rem">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
                        <span style="font-family:var(--mono);font-size:0.65rem;color:var(--text)">{sn}</span>
                        <span style="font-family:var(--mono);font-size:0.6rem;color:{sc}">{sv*100:.0f}%</span>
                    </div>
                    <div class="bar-bg"><div class="bar-fill" style="width:{sv*100:.0f}%;background:{sc}"></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="info-row">AI phrase hits: <span class="iv">{r["phrase_hits"]}</span> &nbsp;·&nbsp; Word count: <span class="iv">{r["word_count"]}</span></div>', unsafe_allow_html=True)
            if iai:
                st.markdown('<div class="card-sm" style="border-left:3px solid #a855f7;margin-top:0.4rem"><div style="font-family:var(--sans);font-size:0.76rem;color:var(--text);line-height:1.7">⚠ Caption shows AI-generation patterns — formulaic phrasing or verbose structure detected. This is an additional forgery signal.</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PRESENTATION STATS ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-label">Model Performance — Project Overview</div>', unsafe_allow_html=True)

    sc = st.columns(6)
    for col, (val, lbl) in zip(sc, [
        ("93.1%","TEST ACCURACY"),("0.9285","F1 SCORE"),
        ("0.9798","AUC-ROC"),    ("0.9425","BEST VAL F1"),
        ("94.44%","BEST VAL ACC"),("14","BEST EPOCH")
    ]):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ac = st.columns(5)
    for col, (t, d) in zip(ac, [
        ("👁 Vision",   "EfficientNet-B4\nImageNet pretrained\nLast 2 blocks tuned"),
        ("📝 Text",     "DistilBERT-base\n128 max tokens\nLast block tuned"),
        ("🔀 Fusion",   "Cross-Attention\n8 heads · 512-dim\nVision queries text"),
        ("🗺 Localize", "U-Net Decoder\n5 blocks · 7→224px\nPixel heatmap"),
        ("🔍 Extra",    "ELA+DCT+CLIP\nCaption AI detect\n5-layer pipeline"),
    ]):
        with col:
            lh = ''.join(f'<div>{l}</div>' for l in d.split('\n'))
            st.markdown(f'<div class="card-sm"><div style="font-family:var(--mono);font-size:0.63rem;font-weight:600;color:var(--accent);margin-bottom:0.45rem">{t}</div><div style="font-family:var(--mono);font-size:0.58rem;color:var(--text-dim);line-height:1.9">{lh}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;
                padding:1rem 1.5rem;font-family:var(--mono);font-size:0.62rem;color:var(--text-dim);line-height:2">
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2rem">
            <div><span style="color:var(--accent);font-weight:600">DATASET · </span>Flickr8k · 8,091 images · SBI augmentation · Caption perturbation</div>
            <div><span style="color:var(--accent);font-weight:600">TRAINING · </span>Colab T4 · AdamW lr=2e-4 · CosineAnnealingLR · Focal+Dice loss</div>
            <div><span style="color:var(--accent);font-weight:600">PIPELINE · </span>L1 SBI · L2 CLIP · L3 DCT · L4 ELA · L5 Caption-AI</div>
        </div>
    </div>""", unsafe_allow_html=True)


# =============================================================
# FOOTER
# =============================================================
st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);
            font-family:var(--mono);font-size:0.57rem;color:var(--text-dim);
            display:flex;justify-content:space-between">
    <span>SENTINEL</span>
    <span>EfficientNet-B4 · DistilBERT · CLIP ViT-B/32 · ELA · DCT · 5-Layer Pipeline</span>
    <span>F1: 0.9285 · AUC-ROC: 0.9798</span>
</div>""", unsafe_allow_html=True)