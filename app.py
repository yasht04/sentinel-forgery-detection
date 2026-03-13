# =============================================================
# app.py — SENTINEL v4.0
# Changes from v3:
#   1. Simplified UI with dark/light mode toggle
#   2. Per-region fake highlighting (fake bg, real fg separated)
#   3. AI caption detection
#   4. Photoshop + AI forgery detection (separate verdicts)
# Run: streamlit run app.py
# =============================================================

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
    initial_sidebar_state="expanded"
)

# =============================================================
# THEME STATE
# =============================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'light_mode' not in st.session_state:
    st.session_state.light_mode = False

# =============================================================
# CSS
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:          #0e1117;
    --bg2:         #161b25;
    --bg3:         #1c2333;
    --border:      #252f42;
    --border2:     #2e3d57;
    --text:        #d4dff0;
    --text-dim:    #546a87;
    --accent:      #3b82f6;
    --accent-dim:  #3b82f620;
    --red:         #ef4444;
    --red-dim:     #ef444418;
    --green:       #22c55e;
    --green-dim:   #22c55e18;
    --amber:       #f59e0b;
    --amber-dim:   #f59e0b18;
    --purple:      #a855f7;
    --purple-dim:  #a855f718;
    --mono:        'DM Mono', monospace;
    --sans:        'DM Sans', sans-serif;
    --r:           10px;
    --shadow:      0 4px 20px rgba(0,0,0,0.35);
}

[data-theme="light"] {
    --bg:          #f8fafc;
    --bg2:         #ffffff;
    --bg3:         #f1f5f9;
    --border:      #e2e8f0;
    --border2:     #cbd5e1;
    --text:        #1e293b;
    --text-dim:    #64748b;
    --accent:      #2563eb;
    --accent-dim:  #2563eb12;
    --red:         #dc2626;
    --red-dim:     #dc262612;
    --green:       #16a34a;
    --green-dim:   #16a34a12;
    --amber:       #d97706;
    --amber-dim:   #d9770612;
    --purple:      #9333ea;
    --purple-dim:  #9333ea12;
    --shadow:      0 4px 20px rgba(0,0,0,0.07);
}

* { box-sizing: border-box; }
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--sans) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }

/* Target sidebar text precisely — NO wildcard * which breaks toggle */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div.stMarkdown,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: var(--text) !important; }

/* Sidebar selectbox (theme picker) — always readable */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: var(--bg3) !important;
    border: 1.5px solid var(--accent) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg { fill: var(--accent) !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] span { color: var(--text) !important; }
[data-testid="stSidebar"] [data-testid="stToggle"] { display: flex !important; align-items: center !important; }
[data-testid="stSidebar"] [data-testid="stToggle"] label { color: var(--text) !important; font-size: 0.9rem !important; }
[data-testid="stSidebar"] [data-testid="stToggle"] [role="switch"] {
    background-color: var(--border2) !important;
    border: 2px solid var(--accent) !important;
    min-width: 44px !important;
    height: 24px !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background-color: var(--accent) !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] [role="switch"]::before {
    background: white !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
}

.s-header { display:flex; align-items:center; justify-content:space-between; padding:1.2rem 0 1rem; border-bottom:1px solid var(--border); margin-bottom:1.5rem; }
.s-logo-row { display:flex; align-items:center; gap:1rem; }
.s-logo { width:44px; height:44px; background:var(--accent-dim); border:1.5px solid var(--accent); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; }
.s-title { font-size:1.5rem; font-weight:700; color:var(--text); letter-spacing:-0.02em; }
.s-subtitle { font-family:var(--mono); font-size:0.62rem; color:var(--text-dim); margin-top:0.15rem; }
.s-badge { font-family:var(--mono); font-size:0.6rem; padding:0.3rem 0.8rem; border-radius:20px; border:1px solid var(--green); color:var(--green); background:var(--green-dim); }

.sec-label { font-family:var(--mono); font-size:0.6rem; letter-spacing:0.12em; color:var(--text-dim); text-transform:uppercase; margin-bottom:0.8rem; display:flex; align-items:center; gap:0.6rem; }
.sec-label::after { content:''; flex:1; height:1px; background:var(--border); }

.card { background:var(--bg2); border:1px solid var(--border); border-radius:var(--r); padding:1.2rem; box-shadow:var(--shadow); }
.card-sm { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:0.85rem 1rem; }

.verdict-wrap { border-radius:var(--r); padding:1.3rem; text-align:center; border:1.5px solid; }
.verdict-authentic { border-color:var(--green);  background:var(--green-dim); }
.verdict-ai        { border-color:var(--purple); background:var(--purple-dim); }
.verdict-photoshop { border-color:var(--amber);  background:var(--amber-dim); }
.verdict-spliced   { border-color:var(--red);    background:var(--red-dim); }
.verdict-manip     { border-color:var(--amber);  background:var(--amber-dim); }
.verdict-icon  { font-size:1.8rem; margin-bottom:0.3rem; }
.verdict-label { font-size:1.3rem; font-weight:700; }
.verdict-desc  { font-family:var(--mono); font-size:0.62rem; color:var(--text-dim); margin-top:0.35rem; }
.vc-green  { color:var(--green); }
.vc-purple { color:var(--purple); }
.vc-amber  { color:var(--amber); }
.vc-red    { color:var(--red); }

.pill-row { display:grid; grid-template-columns:repeat(3,1fr); gap:0.6rem; margin:0.8rem 0; }
.pill { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:0.75rem 0.5rem; text-align:center; }
.pill-val { font-family:var(--mono); font-size:1.4rem; font-weight:500; line-height:1; }
.pill-lbl { font-family:var(--mono); font-size:0.52rem; color:var(--text-dim); letter-spacing:0.08em; margin-top:0.3rem; text-transform:uppercase; }

.layer-row { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.45rem; display:flex; align-items:center; gap:0.85rem; border-left:3px solid var(--border2); }
.layer-row-green  { border-left-color:var(--green); }
.layer-row-amber  { border-left-color:var(--amber); }
.layer-row-red    { border-left-color:var(--red); }
.layer-row-purple { border-left-color:var(--purple); }
.layer-row-dim    { border-left-color:var(--border2); opacity:0.5; }
.layer-icon  { font-size:1.1rem; flex-shrink:0; }
.layer-info  { flex:1; }
.layer-name  { font-size:0.82rem; font-weight:600; color:var(--text); }
.layer-desc  { font-family:var(--mono); font-size:0.58rem; color:var(--text-dim); margin-top:0.1rem; }
.layer-right { text-align:right; }
.layer-score { font-family:var(--mono); font-size:1.1rem; font-weight:500; }
.layer-risk  { font-family:var(--mono); font-size:0.56rem; letter-spacing:0.06em; }

.bar-bg   { background:var(--border); border-radius:3px; height:3px; margin-top:0.4rem; overflow:hidden; }
.bar-fill { height:100%; border-radius:3px; }

[data-testid="stFileUploader"] { border:1.5px dashed var(--border2) !important; border-radius:var(--r) !important; background:var(--bg2) !important; }
[data-testid="stFileUploader"]:hover { border-color:var(--accent) !important; }

.stButton > button { background:var(--accent) !important; border:none !important; color:#fff !important; font-family:var(--sans) !important; font-size:0.85rem !important; font-weight:600 !important; padding:0.65rem 1.5rem !important; border-radius:8px !important; letter-spacing:0.01em !important; }
.stButton > button:hover { opacity:0.88 !important; }
.stButton > button:disabled { background:var(--border2) !important; opacity:0.5 !important; }

.stTabs [data-baseweb="tab-list"] { background:var(--bg2) !important; border-bottom:1px solid var(--border) !important; gap:0 !important; }
.stTabs [data-baseweb="tab"] { font-family:var(--mono) !important; font-size:0.65rem !important; letter-spacing:0.06em !important; color:var(--text-dim) !important; padding:0.5rem 0.9rem !important; }
.stTabs [aria-selected="true"] { color:var(--accent) !important; border-bottom:2px solid var(--accent) !important; background:var(--accent-dim) !important; }

.sb-row { display:flex; justify-content:space-between; align-items:center; padding:0.45rem 0; border-bottom:1px solid var(--border); font-family:var(--mono); font-size:0.65rem; color:var(--text-dim); }
.sb-val { color:var(--accent); font-weight:500; }

textarea { background:var(--bg2) !important; border:1px solid var(--border2) !important; color:var(--text) !important; font-family:var(--mono) !important; font-size:0.78rem !important; border-radius:8px !important; }

.empty-state { border:1.5px dashed var(--border2); border-radius:var(--r); padding:3rem 2rem; text-align:center; }
.empty-icon  { font-size:2.2rem; opacity:0.3; }
.empty-text  { font-family:var(--mono); font-size:0.7rem; color:var(--text-dim); margin-top:0.8rem; line-height:1.9; }

.info-row { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:0.8rem 1rem; font-family:var(--mono); font-size:0.63rem; color:var(--text-dim); line-height:2; }
.iv { color:var(--text); }

.rbox { border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.45rem; border:1px solid var(--border); }
.rbox-real   { border-left:3px solid var(--green);  background:var(--green-dim); }
.rbox-fake   { border-left:3px solid var(--red);    background:var(--red-dim); }
.rbox-ai     { border-left:3px solid var(--purple); background:var(--purple-dim); }
.rbox-neutral{ border-left:3px solid var(--border2); }

.stat-card { background:var(--bg2); border:1px solid var(--border); border-top:2px solid var(--accent); border-radius:8px; padding:1rem 0.6rem; text-align:center; }
.stat-val  { font-family:var(--mono); font-size:1.25rem; font-weight:500; color:var(--accent); }
.stat-lbl  { font-family:var(--mono); font-size:0.52rem; color:var(--text-dim); letter-spacing:0.08em; margin-top:0.25rem; text-transform:uppercase; }

.stSpinner > div { border-top-color:var(--accent) !important; }
[data-testid="stSlider"] [role="slider"] { background:var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# Inject theme attribute
st.markdown(
    f'<script>document.documentElement.setAttribute("data-theme","{st.session_state.theme}")</script>',
    unsafe_allow_html=True
)


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


def clip_ai_detection(image_pil, cm, cp):
    try:
        real_p = ["a real photograph taken by a camera",
                  "a natural photo with realistic lighting",
                  "a genuine photograph of a real scene",
                  "a photo taken with a smartphone"]
        ai_p   = ["an AI generated image by artificial intelligence",
                  "a fake image from a GAN or diffusion model",
                  "an artificially generated synthetic image",
                  "an image created by MidJourney or DALL-E",
                  "a digital artwork made by a neural network"]
        inputs = cp(text=real_p+ai_p, images=image_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            pr = torch.softmax(cm(**inputs).logits_per_image.squeeze(0), dim=0)
        rs = pr[:len(real_p)].sum().item()
        as_ = pr[len(real_p):].sum().item()
        return float(as_ / (as_ + rs + 1e-8)), True
    except Exception:
        return 0.5, False


def detect_ai_caption(caption, cm, cp, clip_ok):
    """Detect if caption was written by AI using heuristics + CLIP text features."""
    if not caption or not caption.strip():
        return None
    text = caption.strip()
    ai_phrases = ['showcasing','depicting','highlighting','featuring',
                  'in this image','in the image','the image shows','the image depicts',
                  'illustrating','demonstrating','capturing the essence',
                  'a stunning','a breathtaking','this photograph','this photo shows',
                  'set against','bathed in','surrounded by','adorned with']
    tl = text.lower()
    hits       = sum(1 for p in ai_phrases if p in tl)
    wc         = len(text.split())
    avg_wl     = np.mean([len(w) for w in text.split()]) if text.split() else 0
    len_sig    = 1.0 if 12 <= wc <= 40 else 0.3
    phrase_sig = min(1.0, hits * 0.35)
    vocab_sig  = min(1.0, max(0.0, (avg_wl - 4.5) / 3.0))
    h_score    = 0.40*phrase_sig + 0.35*len_sig*0.5 + 0.25*vocab_sig

    clip_sig = 0.5
    if clip_ok and cm is not None:
        try:
            prompts = [f'A human casually wrote: "{text[:100]}"',
                       f'An AI language model generated: "{text[:100]}"']
            inp = cp(text=prompts, return_tensors="pt", padding=True)
            with torch.no_grad():
                tf = F.normalize(cm.get_text_features(**inp), dim=-1)
                sim = float(torch.dot(tf[0], tf[1]).item())
                clip_sig = max(0.0, min(1.0, (0.95 - sim) * 5.0))
        except Exception:
            clip_sig = 0.5

    score = max(0.0, min(1.0, 0.60*h_score + 0.40*clip_sig))
    label = 'AI-GENERATED' if score > 0.50 else 'HUMAN-WRITTEN'
    conf  = score if score > 0.50 else (1.0 - score)
    return {'score': score, 'label': label, 'confidence': conf,
            'phrase_hits': hits, 'word_count': wc, 'heuristic': h_score, 'clip_signal': clip_sig}


@st.cache_resource
def load_sbi_model():
    device    = torch.device('cpu')
    model     = ForgeryDetector()
    weights   = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(weights['model_state'])
    model.eval()
    tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tok, device


def run_sbi_inference(image_pil, caption, model, tok, device):
    IMG = 224
    image_np  = cv2.resize(np.array(image_pil.convert('RGB')), (IMG, IMG))
    tf        = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img_t     = tf(image_np).unsqueeze(0).to(device)
    enc       = tok(caption.strip() or "an image", padding='max_length',
                    max_length=128, truncation=True, return_tensors='pt')
    with torch.no_grad():
        logit, heatmap = model(img_t, enc['input_ids'].to(device), enc['attention_mask'].to(device))
    return torch.sigmoid(logit).item(), heatmap.squeeze().cpu().numpy(), image_np


def segment_regions(heatmap_np, image_np, threshold=0.5):
    """Highlight fake regions in color, dim real regions to grayscale."""
    h, w      = image_np.shape[:2]
    hm        = cv2.resize(heatmap_np, (w, h))
    fake_mask = (hm > threshold).astype(np.uint8)
    real_mask = 1 - fake_mask
    contours, _ = cv2.findContours(fake_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray      = np.stack([cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)]*3, axis=2)
    out       = (fake_mask[:,:,None] * image_np.astype(np.float32) +
                 real_mask[:,:,None] * (0.40*image_np.astype(np.float32) + 0.60*gray.astype(np.float32)))
    red_layer = np.zeros_like(out); red_layer[:,:,0] = 255
    out       = out + fake_mask[:,:,None] * red_layer * 0.28
    out       = np.clip(out, 0, 255).astype(np.uint8)
    cv2.drawContours(out, contours, -1, (255, 55, 55), 2)
    fake_pct  = float(fake_mask.mean() * 100)
    n_regions = len([c for c in contours if cv2.contourArea(c) > 50])
    return out, fake_pct, 100.0 - fake_pct, n_regions


def create_overlay(image_np, heatmap_np, alpha=0.45):
    colored = colormaps['jet'](heatmap_np)[:, :, :3]
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


def combine_scores(sbi, clip_s, dct_ai, ela_ps, dct_ps, clip_ok, cap_analysis=None):
    if clip_ok:
        ai  = 0.35*sbi + 0.42*clip_s + 0.23*dct_ai
    else:
        ai  = 0.55*sbi + 0.45*dct_ai
    ps = 0.50*sbi + 0.30*ela_ps + 0.20*dct_ps
    if cap_analysis:
        mm = max(0.0, 0.5 - cap_analysis['match_score'])
        ai = min(1.0, ai + 0.20*mm)
    ai_layers = [sbi, dct_ai] + ([clip_s] if clip_ok else [])
    if sum(1 for s in ai_layers if s > 0.60) >= 2:
        ai = min(1.0, ai + 0.07)
    if sum(1 for s in [sbi, ela_ps, dct_ps] if s > 0.55) >= 2:
        ps = min(1.0, ps + 0.07)
    return float(max(ai, ps)), float(ai), float(ps)


def get_verdict(final, ai, ps, threshold):
    if final < threshold:
        return "AUTHENTIC",    "No manipulation detected",                   "✓", "vc-green",  "verdict-authentic"
    if ai >= ps:
        if ai > 0.70: return "AI GENERATED",  "AI/GAN/diffusion origin detected",   "⚠", "vc-purple", "verdict-ai"
        return               "LIKELY AI",      "Possible AI generation",              "◐", "vc-purple", "verdict-ai"
    else:
        if ps > 0.70: return "PHOTOSHOPPED",  "Photoshop/splice manipulation found", "✗", "vc-red",    "verdict-photoshop"
        return               "MANIPULATED",   "Image has been altered",              "⚠", "vc-amber",  "verdict-manip"


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
# SIDEBAR
# =============================================================
with st.sidebar:
    # ── Theme selector — selectbox is always visible ──────────
    st.markdown(
        '<p style="font-size:0.75rem;font-weight:600;margin-bottom:0.2rem;color:#888">🎨 THEME</p>',
        unsafe_allow_html=True
    )
    theme_choice = st.selectbox(
        "Theme",
        options=["🌙 Dark", "☀️ Light"],
        index=1 if st.session_state.light_mode else 0,
        label_visibility="collapsed"
    )
    chosen = 'light' if '☀' in theme_choice else 'dark'
    if chosen != st.session_state.theme:
        st.session_state.theme      = chosen
        st.session_state.light_mode = (chosen == 'light')
        st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-family:var(--mono);font-size:0.68rem;font-weight:600;color:var(--accent);margin-bottom:0.5rem;letter-spacing:0.06em">SYSTEMS</div>', unsafe_allow_html=True)
    for nm, st_txt, col in [
        ("SBI Splice",     "Online",                           "#22c55e"),
        ("CLIP AI",        "Online" if clip_ok else "Offline", "#22c55e" if clip_ok else "#ef4444"),
        ("DCT Frequency",  "Online",                           "#22c55e"),
        ("ELA Photoshop",  "Online",                           "#22c55e"),
        ("Caption AI Det", "Online",                           "#22c55e"),
    ]:
        st.markdown(f'<div class="sb-row"><span>{nm}</span><span class="sb-val" style="color:{col}">● {st_txt}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:var(--mono);font-size:0.68rem;font-weight:600;color:var(--accent);margin-bottom:0.5rem;letter-spacing:0.06em">MODEL METRICS</div>', unsafe_allow_html=True)
    for lbl, val in [("Accuracy","93.1%"),("F1","0.9285"),("AUC-ROC","0.9798"),("Val F1","0.9425")]:
        st.markdown(f'<div class="sb-row"><span>{lbl}</span><span class="sb-val">{val}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:var(--mono);font-size:0.68rem;font-weight:600;color:var(--accent);margin-bottom:0.3rem;letter-spacing:0.06em">SENSITIVITY</div>', unsafe_allow_html=True)
    threshold = st.slider("Threshold", 0.10, 0.90, 0.45, 0.05, label_visibility="collapsed")
    st.markdown(f'<div style="font-family:var(--mono);font-size:0.6rem;color:var(--text-dim)">Fake threshold: <span style="color:var(--accent)">{threshold:.2f}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:var(--mono);font-size:0.68rem;font-weight:600;color:var(--accent);margin-bottom:0.3rem;letter-spacing:0.06em">REGION THRESHOLD</div>', unsafe_allow_html=True)
    region_thr = st.slider("Region", 0.20, 0.80, 0.50, 0.05, label_visibility="collapsed")
    st.markdown(f'<div style="font-family:var(--mono);font-size:0.6rem;color:var(--text-dim)">Pixel marked fake above <span style="color:var(--accent)">{region_thr:.2f}</span></div>', unsafe_allow_html=True)


# =============================================================
# HEADER
# =============================================================
st.markdown(f"""
<div class="s-header">
    <div class="s-logo-row">
        <div class="s-logo">🛡️</div>
        <div>
            <div class="s-title">SENTINEL</div>
            <div class="s-subtitle">MEDIA FORGERY DETECTION · v4.0</div>
        </div>
    </div>
    <div class="s-badge">{'● ALL SYSTEMS ONLINE' if clip_ok else '● CLIP OFFLINE'}</div>
</div>""", unsafe_allow_html=True)


# =============================================================
# MAIN TWO-COLUMN LAYOUT
# =============================================================
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown('<div class="sec-label">Evidence Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
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

        with st.spinner("L1 — SBI splice detection..."):
            sbi_score, heatmap_np, image_np = run_sbi_inference(image_pil, caption, sbi_model, tokenizer, device)
        with st.spinner("L2 — CLIP AI detection..."):
            clip_score, _ = (clip_ai_detection(image_pil, clip_model, clip_processor) if clip_ok else (0.5, False))
        with st.spinner("L3 — DCT frequency..."):
            dct_ai, dct_ps, dct_d = dct_frequency_analysis(image_pil)
        with st.spinner("L4 — ELA Photoshop..."):
            ela_score, ela_gray, ela_vis = ela_analysis(image_pil)
        with st.spinner("L5 — Caption analysis..."):
            cap_cons   = analyze_caption_consistency(image_pil, caption, clip_model, clip_processor, clip_ok)
            cap_ai_res = detect_ai_caption(caption, clip_model, clip_processor, clip_ok)
        with st.spinner("Computing verdict..."):
            final, ai_score, ps_score = combine_scores(sbi_score, clip_score, dct_ai, ela_score, dct_ps, clip_ok, cap_cons)
            v_lbl, v_desc, v_icon, v_col, v_cls = get_verdict(final, ai_score, ps_score, threshold)

        # Verdict
        st.markdown(f"""
        <div class="verdict-wrap {v_cls}">
            <div class="verdict-icon">{v_icon}</div>
            <div class="verdict-label {v_col}">{v_lbl}</div>
            <div class="verdict-desc">{v_desc} · Confidence {final*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # Score pills
        fc = '#ef4444' if v_lbl != 'AUTHENTIC' else '#22c55e'
        st.markdown(f"""
        <div class="pill-row">
            <div class="pill"><div class="pill-val" style="color:{fc}">{final*100:.1f}%</div><div class="pill-lbl">Fake Score</div></div>
            <div class="pill"><div class="pill-val" style="color:#a855f7">{ai_score*100:.1f}%</div><div class="pill-lbl">AI Score</div></div>
            <div class="pill"><div class="pill-val" style="color:#f59e0b">{ps_score*100:.1f}%</div><div class="pill-lbl">Edit Score</div></div>
        </div><br>""", unsafe_allow_html=True)

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
    <span>SENTINEL v4.0</span>
    <span>EfficientNet-B4 · DistilBERT · CLIP ViT-B/32 · ELA · DCT · 5-Layer Pipeline</span>
    <span>F1: 0.9285 · AUC-ROC: 0.9798</span>
</div>""", unsafe_allow_html=True)