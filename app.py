# # =============================================================
# # app.py — Intelligent Media Forgery Detection (v3)
# # Modernized UI: Forensic Lab / Cyberpunk Terminal aesthetic
# # Run with: streamlit run app.py
# # =============================================================

# import streamlit as st
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# import numpy as np
# import cv2
# import matplotlib
# from matplotlib import colormaps
# from PIL import Image
# import sys, os
# from scipy.fftpack import dct as scipy_dct

# sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
# from architecture import ForgeryDetector
# from transformers import DistilBertTokenizer

# # =============================================================
# # PAGE CONFIG
# # =============================================================
# st.set_page_config(
#     page_title="SENTINEL — Forgery Detection",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # =============================================================
# # CSS — FORENSIC LAB / CYBERPUNK TERMINAL
# # =============================================================
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@700;800&display=swap');

# /* ── Base ─────────────────────────────────────────────── */
# :root {
#     --bg:        #080b10;
#     --bg2:       #0d1117;
#     --bg3:       #111820;
#     --border:    #1e2d3d;
#     --amber:     #f0a500;
#     --amber-dim: #f0a50044;
#     --amber-glow:#f0a50022;
#     --red:       #ff3b3b;
#     --red-dim:   #ff3b3b33;
#     --green:     #00e676;
#     --green-dim: #00e67622;
#     --cyan:      #00b4d8;
#     --text:      #c8d6e5;
#     --text-dim:  #5a7a8a;
#     --mono:      'Share Tech Mono', monospace;
#     --sans:      'Barlow', sans-serif;
#     --display:   'Barlow Condensed', sans-serif;
# }

# * { box-sizing: border-box; }

# .stApp {
#     background-color: var(--bg);
#     color: var(--text);
#     font-family: var(--sans);
# }

# /* Scanline overlay */
# .stApp::before {
#     content: '';
#     position: fixed;
#     top: 0; left: 0;
#     width: 100%; height: 100%;
#     background: repeating-linear-gradient(
#         0deg,
#         transparent,
#         transparent 2px,
#         rgba(0,0,0,0.03) 2px,
#         rgba(0,0,0,0.03) 4px
#     );
#     pointer-events: none;
#     z-index: 9999;
# }

# /* ── Hide Streamlit chrome ────────────────────────────── */
# #MainMenu, footer, header { visibility: hidden; }
# .block-container { padding-top: 1.5rem !important; }
# [data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border); }

# /* ── Header ───────────────────────────────────────────── */
# .sentinel-header {
#     display: flex;
#     align-items: center;
#     gap: 1.2rem;
#     padding: 1.5rem 0 0.5rem;
#     border-bottom: 1px solid var(--border);
#     margin-bottom: 1.5rem;
#     position: relative;
# }
# .sentinel-logo {
#     width: 52px; height: 52px;
#     border: 2px solid var(--amber);
#     border-radius: 10px;
#     display: flex; align-items: center; justify-content: center;
#     font-size: 1.6rem;
#     background: var(--amber-glow);
#     box-shadow: 0 0 20px var(--amber-dim);
#     flex-shrink: 0;
# }
# .sentinel-title {
#     font-family: var(--display);
#     font-size: 2.4rem;
#     font-weight: 800;
#     letter-spacing: 0.15em;
#     color: var(--amber);
#     text-shadow: 0 0 30px var(--amber-dim);
#     line-height: 1;
# }
# .sentinel-sub {
#     font-family: var(--mono);
#     font-size: 0.7rem;
#     color: var(--text-dim);
#     letter-spacing: 0.12em;
#     margin-top: 0.3rem;
# }
# .sentinel-badge {
#     margin-left: auto;
#     font-family: var(--mono);
#     font-size: 0.65rem;
#     color: var(--green);
#     background: var(--green-dim);
#     border: 1px solid var(--green);
#     padding: 0.3rem 0.7rem;
#     border-radius: 4px;
#     letter-spacing: 0.08em;
# }

# /* ── Panel base ────────────────────────────────────────── */
# .panel {
#     background: var(--bg3);
#     border: 1px solid var(--border);
#     border-radius: 8px;
#     padding: 1.4rem;
#     position: relative;
#     overflow: hidden;
# }
# .panel::before {
#     content: '';
#     position: absolute;
#     top: 0; left: 0; right: 0;
#     height: 2px;
#     background: linear-gradient(90deg, transparent, var(--amber), transparent);
#     opacity: 0.6;
# }

# /* ── Section label ─────────────────────────────────────── */
# .sec-label {
#     font-family: var(--mono);
#     font-size: 0.65rem;
#     letter-spacing: 0.18em;
#     color: var(--amber);
#     text-transform: uppercase;
#     margin-bottom: 1rem;
#     display: flex;
#     align-items: center;
#     gap: 0.5rem;
# }
# .sec-label::after {
#     content: '';
#     flex: 1;
#     height: 1px;
#     background: var(--border);
# }

# /* ── Verdict boxes ─────────────────────────────────────── */
# .verdict-box {
#     border-radius: 8px;
#     padding: 1.6rem;
#     text-align: center;
#     position: relative;
#     overflow: hidden;
# }
# .verdict-authentic {
#     background: linear-gradient(135deg, #00e67608, #00e67603);
#     border: 1px solid var(--green);
#     box-shadow: 0 0 30px #00e67615, inset 0 0 30px #00e67605;
# }
# .verdict-ai {
#     background: linear-gradient(135deg, #f0a50008, #f0a50003);
#     border: 1px solid var(--amber);
#     box-shadow: 0 0 30px var(--amber-dim), inset 0 0 30px var(--amber-glow);
# }
# .verdict-fake {
#     background: linear-gradient(135deg, #ff3b3b08, #ff3b3b03);
#     border: 1px solid var(--red);
#     box-shadow: 0 0 30px #ff3b3b22, inset 0 0 30px #ff3b3b08;
# }
# .verdict-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
# .verdict-label {
#     font-family: var(--display);
#     font-size: 1.9rem;
#     font-weight: 800;
#     letter-spacing: 0.12em;
# }
# .verdict-label-auth  { color: var(--green); text-shadow: 0 0 20px #00e67655; }
# .verdict-label-ai    { color: var(--amber); text-shadow: 0 0 20px var(--amber-dim); }
# .verdict-label-fake  { color: var(--red);   text-shadow: 0 0 20px #ff3b3b55; }
# .verdict-desc {
#     font-family: var(--mono);
#     font-size: 0.7rem;
#     color: var(--text-dim);
#     margin-top: 0.5rem;
#     letter-spacing: 0.08em;
# }

# /* ── Score cards ────────────────────────────────────────── */
# .score-grid {
#     display: grid;
#     grid-template-columns: repeat(3, 1fr);
#     gap: 0.8rem;
#     margin-top: 1rem;
# }
# .score-card {
#     background: var(--bg2);
#     border: 1px solid var(--border);
#     border-radius: 6px;
#     padding: 1rem 0.8rem;
#     text-align: center;
# }
# .score-num {
#     font-family: var(--mono);
#     font-size: 1.7rem;
#     font-weight: 700;
#     line-height: 1;
# }
# .score-lbl {
#     font-family: var(--mono);
#     font-size: 0.58rem;
#     color: var(--text-dim);
#     letter-spacing: 0.1em;
#     margin-top: 0.4rem;
#     text-transform: uppercase;
# }

# /* ── Layer cards ────────────────────────────────────────── */
# .layer-row {
#     background: var(--bg2);
#     border: 1px solid var(--border);
#     border-radius: 6px;
#     padding: 1rem 1.1rem;
#     margin-bottom: 0.6rem;
#     display: flex;
#     align-items: center;
#     gap: 1rem;
#     position: relative;
#     overflow: hidden;
# }
# .layer-row::before {
#     content: '';
#     position: absolute;
#     left: 0; top: 0; bottom: 0;
#     width: 3px;
# }
# .layer-row-green::before  { background: var(--green); }
# .layer-row-amber::before  { background: var(--amber); }
# .layer-row-red::before    { background: var(--red);   }
# .layer-row-dim::before    { background: var(--border);}

# .layer-icon { font-size: 1.3rem; flex-shrink: 0; }
# .layer-info { flex: 1; }
# .layer-name {
#     font-family: var(--sans);
#     font-size: 0.88rem;
#     font-weight: 600;
#     color: var(--text);
# }
# .layer-desc {
#     font-family: var(--mono);
#     font-size: 0.62rem;
#     color: var(--text-dim);
#     margin-top: 0.15rem;
# }
# .layer-score-wrap { text-align: right; }
# .layer-score {
#     font-family: var(--mono);
#     font-size: 1.3rem;
#     font-weight: 700;
# }
# .layer-risk {
#     font-family: var(--mono);
#     font-size: 0.6rem;
#     letter-spacing: 0.08em;
# }

# /* Progress bar */
# .bar-bg {
#     background: var(--border);
#     border-radius: 2px;
#     height: 4px;
#     margin-top: 0.5rem;
#     overflow: hidden;
# }
# .bar-fill {
#     height: 100%;
#     border-radius: 2px;
#     transition: width 0.4s ease;
# }

# /* ── Upload zone ────────────────────────────────────────── */
# [data-testid="stFileUploader"] {
#     border: 1px dashed var(--border) !important;
#     border-radius: 8px !important;
#     background: var(--bg2) !important;
#     transition: border-color 0.2s;
# }
# [data-testid="stFileUploader"]:hover {
#     border-color: var(--amber) !important;
# }

# /* ── Buttons ────────────────────────────────────────────── */
# .stButton > button {
#     background: transparent !important;
#     border: 1px solid var(--amber) !important;
#     color: var(--amber) !important;
#     font-family: var(--mono) !important;
#     font-size: 0.8rem !important;
#     letter-spacing: 0.12em !important;
#     padding: 0.7rem 1.5rem !important;
#     border-radius: 4px !important;
#     transition: all 0.2s !important;
# }
# .stButton > button:hover {
#     background: var(--amber-glow) !important;
#     box-shadow: 0 0 20px var(--amber-dim) !important;
# }
# .stButton > button:disabled {
#     border-color: var(--border) !important;
#     color: var(--text-dim) !important;
#     opacity: 0.5 !important;
# }

# /* ── Tabs ────────────────────────────────────────────────── */
# .stTabs [data-baseweb="tab-list"] {
#     background: var(--bg2) !important;
#     border-bottom: 1px solid var(--border) !important;
#     gap: 0 !important;
# }
# .stTabs [data-baseweb="tab"] {
#     font-family: var(--mono) !important;
#     font-size: 0.7rem !important;
#     letter-spacing: 0.1em !important;
#     color: var(--text-dim) !important;
#     padding: 0.6rem 1.2rem !important;
# }
# .stTabs [aria-selected="true"] {
#     color: var(--amber) !important;
#     border-bottom: 2px solid var(--amber) !important;
#     background: var(--amber-glow) !important;
# }

# /* ── Sidebar ────────────────────────────────────────────── */
# .sidebar-title {
#     font-family: var(--display);
#     font-size: 1.1rem;
#     letter-spacing: 0.15em;
#     color: var(--amber);
#     margin-bottom: 0.3rem;
# }
# .sidebar-metric {
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     padding: 0.5rem 0;
#     border-bottom: 1px solid var(--border);
#     font-family: var(--mono);
#     font-size: 0.72rem;
# }
# .sidebar-metric-val { color: var(--amber); font-weight: 700; }

# /* ── Info / empty state ─────────────────────────────────── */
# .empty-state {
#     border: 1px dashed var(--border);
#     border-radius: 8px;
#     padding: 3.5rem 2rem;
#     text-align: center;
# }
# .empty-icon { font-size: 2.8rem; opacity: 0.4; }
# .empty-text {
#     font-family: var(--mono);
#     font-size: 0.75rem;
#     color: var(--text-dim);
#     margin-top: 1rem;
#     line-height: 1.8;
#     letter-spacing: 0.05em;
# }

# /* ── Expander ───────────────────────────────────────────── */
# [data-testid="stExpander"] {
#     background: var(--bg2) !important;
#     border: 1px solid var(--border) !important;
#     border-radius: 6px !important;
# }

# /* ── Alerts ─────────────────────────────────────────────── */
# .stAlert {
#     font-family: var(--mono) !important;
#     font-size: 0.75rem !important;
#     border-radius: 6px !important;
# }

# /* ── Slider ─────────────────────────────────────────────── */
# [data-testid="stSlider"] [data-testid="stThumbValue"] {
#     font-family: var(--mono) !important;
#     font-size: 0.7rem !important;
#     color: var(--amber) !important;
# }

# /* ── Textarea / text input ──────────────────────────────── */
# textarea, input[type="text"] {
#     background: var(--bg2) !important;
#     border: 1px solid var(--border) !important;
#     color: var(--text) !important;
#     font-family: var(--mono) !important;
#     font-size: 0.78rem !important;
#     border-radius: 6px !important;
# }
# textarea:focus, input[type="text"]:focus {
#     border-color: var(--amber) !important;
#     box-shadow: 0 0 0 1px var(--amber-dim) !important;
# }

# /* ── Metrics (native) ───────────────────────────────────── */
# [data-testid="metric-container"] {
#     background: var(--bg2) !important;
#     border: 1px solid var(--border) !important;
#     border-radius: 6px !important;
#     padding: 0.8rem !important;
# }
# [data-testid="stMetricValue"] {
#     font-family: var(--mono) !important;
#     color: var(--amber) !important;
# }
# [data-testid="stMetricLabel"] {
#     font-family: var(--mono) !important;
#     font-size: 0.65rem !important;
#     color: var(--text-dim) !important;
# }

# /* ── Spinner ─────────────────────────────────────────────── */
# .stSpinner > div {
#     border-top-color: var(--amber) !important;
# }
# </style>
# """, unsafe_allow_html=True)


# # =============================================================
# # DETECTION LOGIC (unchanged from v2)
# # =============================================================

# def dct_frequency_analysis(image_pil):
#     img_gray   = np.array(image_pil.convert('L')).astype(np.float32)
#     img_gray   = cv2.resize(img_gray, (256, 256))
#     dct_coeffs = scipy_dct(scipy_dct(img_gray, axis=0, norm='ortho'), axis=1, norm='ortho')
#     dct_log    = np.log(np.abs(dct_coeffs) + 1e-8)
#     h, w       = dct_log.shape
#     hf_energy  = np.mean(np.abs(dct_log[h//2:, w//2:]))
#     lf_energy  = np.mean(np.abs(dct_log[:h//4, :w//4]))
#     hf_ratio   = hf_energy / (lf_energy + 1e-8)
#     dct_abs    = np.abs(dct_coeffs)
#     peak_ratio = np.sum(dct_abs > np.percentile(dct_abs, 99)) / dct_abs.size
#     noise_floor= np.std(dct_log[h//2:, w//2:])
#     hf_score   = max(0, min(1, 1 - (hf_ratio / 0.15)))
#     peak_score = max(0, min(1, peak_ratio / 0.005))
#     noise_score= max(0, min(1, 1 - (noise_floor / 2.5)))
#     ai_score   = 0.45*hf_score + 0.35*peak_score + 0.20*noise_score
#     smoothness = 1.0 / (np.std(np.array(image_pil.convert('RGB')).astype(np.float32)) / 50.0 + 1.0)
#     ai_score   = min(1.0, ai_score + 0.15 * smoothness)
#     return float(ai_score), {'hf_ratio': float(hf_ratio), 'peak_ratio': float(peak_ratio),
#                               'noise_floor': float(noise_floor), 'smoothness': float(smoothness)}


# @st.cache_resource
# def load_clip_model():
#     try:
#         from transformers import CLIPProcessor, CLIPModel
#         model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         model.eval()
#         return model, processor, True
#     except Exception:
#         return None, None, False


# def clip_ai_detection(image_pil, clip_model, clip_processor):
#     try:
#         real_prompts = [
#             "a real photograph taken by a camera",
#             "a natural photo with realistic lighting and shadows",
#             "a genuine photograph of a real scene",
#             "a photo taken by a person with a smartphone",
#         ]
#         ai_prompts = [
#             "an AI generated image created by artificial intelligence",
#             "a fake image generated by a GAN or diffusion model",
#             "an artificially generated synthetic image",
#             "an image created by MidJourney or DALL-E or Stable Diffusion",
#             "a digital artwork created by a neural network",
#         ]
#         inputs = clip_processor(text=real_prompts + ai_prompts,
#                                 images=image_pil, return_tensors="pt", padding=True)
#         with torch.no_grad():
#             outputs = clip_model(**inputs)
#             probs   = torch.softmax(outputs.logits_per_image.squeeze(0), dim=0)
#         real_score = probs[:len(real_prompts)].sum().item()
#         ai_score   = probs[len(real_prompts):].sum().item()
#         return float(ai_score / (ai_score + real_score + 1e-8)), True
#     except Exception:
#         return 0.5, False


# @st.cache_resource
# def load_sbi_model():
#     device    = torch.device('cpu')
#     model     = ForgeryDetector()
#     weights   = torch.load('best_model.pth', map_location=device)
#     model.load_state_dict(weights['model_state'])
#     model.eval()
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#     return model, tokenizer, device


# def run_sbi_inference(image_pil, caption, model, tokenizer, device):
#     IMG_SIZE   = 224
#     image_np   = cv2.resize(np.array(image_pil.convert('RGB')), (IMG_SIZE, IMG_SIZE))
#     transform  = T.Compose([T.ToTensor(),
#                              T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
#     img_tensor = transform(image_np).unsqueeze(0).to(device)
#     enc        = tokenizer(caption.strip() or "an image", padding='max_length',
#                            max_length=128, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         logit, heatmap = model(img_tensor, enc['input_ids'].to(device),
#                                enc['attention_mask'].to(device))
#     return torch.sigmoid(logit).item(), heatmap.squeeze().cpu().numpy(), image_np


# def analyze_caption_image_consistency(image_pil, caption, clip_model, clip_processor, clip_ok):
#     """
#     Measures how well the caption matches the image using CLIP.

#     FIX: We use softmax over [caption vs random descriptions] rather than
#     sigmoid on a raw logit — sigmoid always returns ~1.0 because CLIP logits
#     are large by design (temperature-scaled). Softmax gives a proper relative score.

#     High score = caption matches image (consistent)
#     Low  score = caption contradicts image (mismatch = forgery signal)
#     """
#     if not clip_ok or not caption.strip():
#         return None

#     try:
#         # ── MATCH SCORE ──────────────────────────────────────────────
#         # Compare: user caption vs 4 generic "unrelated" descriptions
#         # Softmax tells us: how much does the image prefer the user caption
#         # over a set of unrelated alternatives?
#         user_caption   = caption.strip()
#         contrast_texts = [
#             user_caption,
#             "a random unrelated image with no specific subject",
#             "an abstract pattern with no recognizable content",
#             "a blank or empty scene",
#             "something completely different from the description",
#         ]
#         inputs = clip_processor(
#             text=contrast_texts,
#             images=image_pil,
#             return_tensors="pt",
#             padding=True
#         )
#         with torch.no_grad():
#             outputs = clip_model(**inputs)
#             # logits_per_image: (1, num_texts) — how well image matches each text
#             logits  = outputs.logits_per_image.squeeze(0)   # (5,)
#             probs   = torch.softmax(logits, dim=0)           # normalized over all 5
#             # prob[0] = how much the image prefers the user caption vs the 4 fillers
#             match_score = float(probs[0].item())

#         # Scale match_score: softmax over 5 options means random = 0.20
#         # Rescale so 0.20 → 0.0 and 1.0 → 1.0 for intuitive display
#         match_score = max(0.0, (match_score - 0.20) / 0.80)
#         match_score = min(1.0, match_score)

#         # ── KEYWORD SCORES ────────────────────────────────────────────
#         # For each meaningful word in caption, ask:
#         # "Does the image contain X?" vs "Does the image NOT contain X?"
#         words       = user_caption.lower().split()
#         # Filter out stopwords for cleaner analysis
#         stopwords   = {'a','an','the','is','are','was','were','be','been',
#                        'has','have','had','do','does','did','will','would',
#                        'can','could','should','may','might','of','in','on',
#                        'at','to','for','with','by','from','and','or','but',
#                        'not','this','that','it','its'}
#         key_words   = [w for w in words if len(w) > 2 and w not in stopwords][:6]
#         word_scores = {}

#         for word in key_words:
#             probe = clip_processor(
#                 text=[f"a photo containing {word}",
#                       f"a photo that does not contain {word}"],
#                 images=image_pil,
#                 return_tensors="pt",
#                 padding=True
#             )
#             with torch.no_grad():
#                 out   = clip_model(**probe)
#                 probs_w = torch.softmax(out.logits_per_image.squeeze(0), dim=0)
#                 word_scores[word] = float(probs_w[0].item())

#         return {
#             'match_score' : match_score,
#             'word_scores' : word_scores,
#             'caption'     : user_caption
#         }
#     except Exception:
#         return None


# def combine_scores(sbi, clip_s, dct, clip_ok):
#     combined = (0.35*sbi + 0.40*clip_s + 0.25*dct) if clip_ok else (0.55*sbi + 0.45*dct)
#     if max(sbi, clip_s if clip_ok else 0, dct) > 0.80:
#         combined = min(1.0, combined + 0.10)
#     return float(combined)


# def get_verdict(sbi, clip_s, dct, final, threshold, clip_ok):
#     if final < threshold:
#         return "AUTHENTIC"
#     scores   = {'SBI': sbi, 'DCT': dct}
#     if clip_ok: scores['CLIP'] = clip_s
#     dominant = max(scores.keys(), key=lambda k: scores[k])
#     if dominant == 'CLIP' and clip_s > 0.6:  return "AI_GENERATED"
#     if dominant == 'SBI'  and sbi   > 0.6:   return "SPLICED"
#     return "MANIPULATED"


# def create_overlay(image_np, heatmap_np, alpha=0.45):
#     colored = colormaps['jet'](heatmap_np)[:, :, :3]
#     overlay = alpha * colored + (1-alpha) * (image_np / 255.0)
#     return np.clip(overlay * 255, 0, 255).astype(np.uint8)


# def layer_color_class(s):
#     if s > 0.65: return 'layer-row-red',   '#ff3b3b', 'HIGH RISK'
#     if s > 0.45: return 'layer-row-amber', '#f0a500', 'ELEVATED'
#     return              'layer-row-green', '#00e676', 'CLEAR'


# # =============================================================
# # LOAD MODELS
# # =============================================================
# with st.spinner("INITIALIZING SENTINEL SYSTEMS..."):
#     sbi_model, tokenizer, device         = load_sbi_model()
#     clip_model, clip_processor, clip_ok  = load_clip_model()


# # =============================================================
# # SIDEBAR
# # =============================================================
# with st.sidebar:
#     st.markdown('<div class="sidebar-title">⚙ SYSTEM STATUS</div>', unsafe_allow_html=True)

#     layers = [
#         ("L1 SBI SPLICE DETECTOR",  "ONLINE", "#00e676"),
#         ("L2 CLIP SEMANTIC ENGINE", "ONLINE" if clip_ok else "OFFLINE",
#          "#00e676" if clip_ok else "#ff3b3b"),
#         ("L3 DCT FREQ ANALYZER",    "ONLINE", "#00e676"),
#     ]
#     for name, status, color in layers:
#         st.markdown(f"""
#         <div class="sidebar-metric">
#             <span>{name}</span>
#             <span class="sidebar-metric-val" style="color:{color}">● {status}</span>
#         </div>""", unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown('<div class="sidebar-title">📊 MODEL STATS</div>', unsafe_allow_html=True)
#     for label, val in [("ACCURACY","93.1%"),("F1 SCORE","0.9285"),("AUC-ROC","0.9798"),
#                        ("TRAIN EPOCHS","14 / 15"),("BEST VAL F1","0.9425")]:
#         st.markdown(f"""
#         <div class="sidebar-metric">
#             <span>{label}</span>
#             <span class="sidebar-metric-val">{val}</span>
#         </div>""", unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown('<div class="sidebar-title">🎚 SENSITIVITY</div>', unsafe_allow_html=True)
#     threshold = st.slider("Detection threshold", 0.10, 0.90, 0.45, 0.05,
#                           label_visibility="collapsed")
#     st.markdown(f"""
#     <div style="font-family:var(--mono);font-size:0.65rem;color:var(--text-dim);margin-top:0.3rem">
#         THRESHOLD: <span style="color:var(--amber)">{threshold:.2f}</span>
#         &nbsp;|&nbsp; {'SENSITIVE' if threshold < 0.4 else 'BALANCED' if threshold < 0.6 else 'CONSERVATIVE'}
#     </div>""", unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown('<div class="sidebar-title">ℹ ABOUT</div>', unsafe_allow_html=True)
#     st.markdown("""
#     <div style="font-family:var(--mono);font-size:0.68rem;color:var(--text-dim);line-height:1.8">
#         SENTINEL detects image forgeries<br>
#         using a three-layer AI pipeline.<br><br>
#         L1 → splice & copy-paste<br>
#         L2 → AI/GAN/diffusion origin<br>
#         L3 → frequency domain artifacts
#     </div>""", unsafe_allow_html=True)


# # =============================================================
# # HEADER
# # =============================================================
# st.markdown(f"""
# <div class="sentinel-header">
#     <div class="sentinel-logo">🛡️</div>
#     <div>
#         <div class="sentinel-title">SENTINEL</div>
#         <div class="sentinel-sub">INTELLIGENT MEDIA FORENSICS SYSTEM — v3.0</div>
#     </div>
#     <div class="sentinel-badge">
#         {'● ALL SYSTEMS ONLINE' if clip_ok else '● L2 OFFLINE — 2/3 LAYERS ACTIVE'}
#     </div>
# </div>
# """, unsafe_allow_html=True)


# # =============================================================
# # MAIN LAYOUT
# # =============================================================
# col_left, col_right = st.columns([1, 1], gap="large")

# # ── LEFT: INPUT ──────────────────────────────────────────────
# with col_left:
#     st.markdown('<div class="sec-label">▸ EVIDENCE INPUT</div>', unsafe_allow_html=True)

#     uploaded_file = st.file_uploader(
#         "DROP IMAGE FILE",
#         type=["jpg","jpeg","png","webp"],
#         label_visibility="collapsed"
#     )
#     st.markdown("""
#     <div style="font-family:var(--mono);font-size:0.62rem;color:var(--text-dim);
#                 margin-top:-0.5rem;margin-bottom:0.8rem;letter-spacing:0.05em">
#         ACCEPTED: JPG · PNG · WEBP · Max size depends on system memory
#     </div>""", unsafe_allow_html=True)

#     caption = st.text_area(
#         "SUBJECT CAPTION",
#         placeholder="Describe image content (optional)...\nEx: a dog running on a beach at sunset",
#         height=90,
#         label_visibility="collapsed"
#     )
#     st.markdown("""
#     <div style="font-family:var(--mono);font-size:0.62rem;color:var(--text-dim);
#                 margin-top:-0.5rem;margin-bottom:1rem;letter-spacing:0.05em">
#         CAPTION IS OPTIONAL — improves semantic mismatch detection
#     </div>""", unsafe_allow_html=True)

#     analyze_btn = st.button(
#         "▶  INITIATE FORENSIC ANALYSIS",
#         use_container_width=True,
#         disabled=(uploaded_file is None)
#     )

#     if uploaded_file is not None:
#         st.markdown("<br>", unsafe_allow_html=True)
#         st.markdown('<div class="sec-label">▸ INPUT PREVIEW</div>', unsafe_allow_html=True)
#         st.image(uploaded_file, use_container_width=True)

#         # File metadata
#         file_bytes = uploaded_file.getvalue()
#         img_check  = Image.open(uploaded_file)
#         st.markdown(f"""
#         <div style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
#                     padding:0.8rem 1rem;margin-top:0.5rem;font-family:var(--mono);
#                     font-size:0.65rem;color:var(--text-dim);line-height:2">
#             FILENAME: <span style="color:var(--text)">{uploaded_file.name}</span><br>
#             DIMENSIONS: <span style="color:var(--text)">{img_check.size[0]} × {img_check.size[1]} px</span><br>
#             FILE SIZE: <span style="color:var(--text)">{len(file_bytes)/1024:.1f} KB</span><br>
#             COLOR MODE: <span style="color:var(--text)">{img_check.mode}</span>
#         </div>""", unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div class="empty-state" style="margin-top:1rem">
#             <div class="empty-icon">📁</div>
#             <div class="empty-text">
#                 NO FILE LOADED<br>
#                 DROP AN IMAGE ABOVE TO BEGIN<br>
#                 FORENSIC ANALYSIS
#             </div>
#         </div>""", unsafe_allow_html=True)


# # ── RIGHT: RESULTS ────────────────────────────────────────────
# with col_right:
#     st.markdown('<div class="sec-label">▸ ANALYSIS REPORT</div>', unsafe_allow_html=True)

#     if uploaded_file is not None and analyze_btn:
#         image_pil = Image.open(uploaded_file)

#         # Run all layers
#         with st.spinner("[ L1 ] RUNNING SBI SPLICE DETECTION..."):
#             sbi_score, heatmap_np, image_np = run_sbi_inference(
#                 image_pil, caption, sbi_model, tokenizer, device
#             )
#         with st.spinner("[ L2 ] RUNNING CLIP SEMANTIC ANALYSIS..."):
#             clip_score, _ = (clip_ai_detection(image_pil, clip_model, clip_processor)
#                              if clip_ok else (0.5, False))
#         with st.spinner("[ L3 ] RUNNING DCT FREQUENCY ANALYSIS..."):
#             dct_score, dct_details = dct_frequency_analysis(image_pil)

#         with st.spinner("[ L4 ] ANALYSING CAPTION-IMAGE CONSISTENCY..."):
#             caption_analysis = analyze_caption_image_consistency(
#                 image_pil, caption, clip_model, clip_processor, clip_ok
#             )

#         final_score = combine_scores(sbi_score, clip_score, dct_score, clip_ok)
#         verdict     = get_verdict(sbi_score, clip_score, dct_score,
#                                   final_score, threshold, clip_ok)

#         # ── VERDICT ──
#         verdict_cfg = {
#             "AUTHENTIC"   : ("verdict-authentic", "verdict-label-auth",  "✓", "AUTHENTIC",    "NO MANIPULATION DETECTED"),
#             "AI_GENERATED": ("verdict-ai",         "verdict-label-ai",   "⚠", "AI GENERATED", "SYNTHETIC ORIGIN DETECTED"),
#             "SPLICED"     : ("verdict-fake",        "verdict-label-fake", "✗", "SPLICED",      "COPY-PASTE FORGERY DETECTED"),
#             "MANIPULATED" : ("verdict-fake",        "verdict-label-fake", "✗", "MANIPULATED",  "TAMPERING DETECTED"),
#         }
#         vbox, vlbl, vicon, vtxt, vdesc = verdict_cfg[verdict]

#         st.markdown(f"""
#         <div class="verdict-box {vbox}">
#             <div class="verdict-icon">{vicon}</div>
#             <div class="verdict-label {vlbl}">{vtxt}</div>
#             <div class="verdict-desc">{vdesc} · CONFIDENCE: {final_score*100:.1f}%</div>
#         </div>""", unsafe_allow_html=True)

#         # ── SCORE CARDS ──
#         fake_color = '#ff3b3b' if verdict != 'AUTHENTIC' else '#00e676'
#         st.markdown(f"""
#         <div class="score-grid">
#             <div class="score-card">
#                 <div class="score-num" style="color:{fake_color}">{final_score*100:.1f}<span style="font-size:0.9rem">%</span></div>
#                 <div class="score-lbl">FAKE SCORE</div>
#             </div>
#             <div class="score-card">
#                 <div class="score-num" style="color:#00e676">{(1-final_score)*100:.1f}<span style="font-size:0.9rem">%</span></div>
#                 <div class="score-lbl">AUTH SCORE</div>
#             </div>
#             <div class="score-card">
#                 <div class="score-num" style="color:#f0a500">{threshold*100:.0f}<span style="font-size:0.9rem">%</span></div>
#                 <div class="score-lbl">THRESHOLD</div>
#             </div>
#         </div>
#         <br>""", unsafe_allow_html=True)

#         # ── LAYER BREAKDOWN ──
#         st.markdown('<div class="sec-label">▸ LAYER ANALYSIS</div>', unsafe_allow_html=True)

#         layers_data = [
#             ("🔬", "L1 — SBI SPLICE DETECTOR",   "Copy-paste & region splicing detection", sbi_score,   True),
#             ("🧠", "L2 — CLIP SEMANTIC ENGINE",   "GAN · Diffusion · MidJourney · DALL-E",  clip_score,  clip_ok),
#             ("📡", "L3 — DCT FREQUENCY ANALYZER", "Unnatural frequency pattern detection",   dct_score,   True),
#         ]

#         for icon, name, desc, score, active in layers_data:
#             if active:
#                 row_cls, score_col, risk_lbl = layer_color_class(score)
#                 bar_w = int(score * 100)
#                 st.markdown(f"""
#                 <div class="layer-row {row_cls}">
#                     <div class="layer-icon">{icon}</div>
#                     <div class="layer-info">
#                         <div class="layer-name">{name}</div>
#                         <div class="layer-desc">{desc}</div>
#                         <div class="bar-bg">
#                             <div class="bar-fill" style="width:{bar_w}%;background:{score_col}"></div>
#                         </div>
#                     </div>
#                     <div class="layer-score-wrap">
#                         <div class="layer-score" style="color:{score_col}">{score*100:.1f}%</div>
#                         <div class="layer-risk"  style="color:{score_col}">{risk_lbl}</div>
#                     </div>
#                 </div>""", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"""
#                 <div class="layer-row layer-row-dim" style="opacity:0.4">
#                     <div class="layer-icon">{icon}</div>
#                     <div class="layer-info">
#                         <div class="layer-name">{name}</div>
#                         <div class="layer-desc" style="color:#ff3b3b">OFFLINE — pip install openai-clip</div>
#                     </div>
#                     <div class="layer-score-wrap">
#                         <div class="layer-score" style="color:var(--border)">N/A</div>
#                     </div>
#                 </div>""", unsafe_allow_html=True)

#         st.markdown("<br>", unsafe_allow_html=True)

#         # ── HEATMAP ──
#         st.markdown('<div class="sec-label">▸ SPLICE LOCALIZATION MAP</div>',
#                     unsafe_allow_html=True)

#         tab1, tab2, tab3 = st.tabs(["OVERLAY", "HEATMAP", "ORIGINAL"])
#         overlay_np = create_overlay(image_np, heatmap_np, alpha=0.45)
#         with tab1:
#             st.image(overlay_np, use_container_width=True)
#             st.markdown("""<div style="font-family:var(--mono);font-size:0.62rem;
#                 color:var(--text-dim);margin-top:0.3rem;text-align:center">
#                 RED/YELLOW REGIONS INDICATE SUSPECTED TAMPERING</div>""",
#                 unsafe_allow_html=True)
#         with tab2:
#             hm_rgb = (colormaps['jet'](heatmap_np)[:,:,:3]*255).astype(np.uint8)
#             st.image(hm_rgb, use_container_width=True)
#         with tab3:
#             st.image(image_np, use_container_width=True)

#         # ── FORENSIC SUMMARY ──
#         st.markdown("<br>", unsafe_allow_html=True)
#         st.markdown('<div class="sec-label">▸ FORENSIC SUMMARY</div>',
#                     unsafe_allow_html=True)

#         forged_pct   = float((heatmap_np > 0.5).mean() * 100)
#         caption_used = caption.strip() if caption.strip() else "(none)"

#         st.markdown(f"""
#         <div style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
#                     padding:1rem 1.2rem;font-family:var(--mono);font-size:0.68rem;
#                     line-height:2.2;color:var(--text-dim)">
#             VERDICT          : <span style="color:var(--amber);font-weight:700">{verdict}</span><br>
#             FINAL FAKE SCORE : <span style="color:var(--text)">{final_score*100:.2f}%</span><br>
#             THRESHOLD USED   : <span style="color:var(--text)">{threshold*100:.0f}%</span><br>
#             SUSPECTED AREA   : <span style="color:var(--text)">{forged_pct:.1f}% of image</span><br>
#             CAPTION INPUT    : <span style="color:var(--text);font-style:italic">{caption_used}</span><br>
#             DCT HF RATIO     : <span style="color:var(--text)">{dct_details['hf_ratio']:.4f}</span><br>
#             DCT NOISE FLOOR  : <span style="color:var(--text)">{dct_details['noise_floor']:.4f}</span>
#         </div>""", unsafe_allow_html=True)

#     else:
#         st.markdown("""
#         <div class="empty-state">
#             <div class="empty-icon">🛡️</div>
#             <div class="empty-text">
#                 SENTINEL STANDING BY<br><br>
#                 UPLOAD AN IMAGE AND CLICK<br>
#                 [ INITIATE FORENSIC ANALYSIS ]<br><br>
#                 DETECTS: AI GENERATED · SPLICED<br>
#                 COPY-PASTE · FREQUENCY ARTIFACTS
#             </div>
#         </div>""", unsafe_allow_html=True)


# # =============================================================
# # BOTTOM PANELS — shown after analysis only
# # =============================================================
# if uploaded_file is not None and analyze_btn:

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("---", unsafe_allow_html=False)

#     # ── PANEL 1: CAPTION-IMAGE CONSISTENCY ──────────────────
#     st.markdown('<div class="sec-label">▸ CAPTION — IMAGE CONSISTENCY ANALYSIS</div>',
#                 unsafe_allow_html=True)

#     if not caption.strip():
#         st.markdown("""
#         <div style="background:var(--bg2);border:1px dashed var(--border);border-radius:8px;
#                     padding:1.5rem;text-align:center;font-family:var(--mono);
#                     font-size:0.72rem;color:var(--text-dim)">
#             NO CAPTION PROVIDED — Enter a description above and re-analyze
#             to see image-caption consistency scoring
#         </div>""", unsafe_allow_html=True)

#     elif not clip_ok:
#         st.markdown("""
#         <div style="background:var(--bg2);border:1px dashed var(--border);border-radius:8px;
#                     padding:1.5rem;text-align:center;font-family:var(--mono);
#                     font-size:0.72rem;color:var(--text-dim)">
#             CLIP NOT AVAILABLE — Caption analysis requires CLIP.<br>
#             Run: pip install openai-clip
#         </div>""", unsafe_allow_html=True)

#     elif caption_analysis:
#         match_score  = caption_analysis['match_score']
#         word_scores  = caption_analysis['word_scores']

#         # Consistency verdict
#         if match_score > 0.65:
#             cons_color = '#00e676'
#             cons_label = 'CONSISTENT'
#             cons_desc  = 'Caption accurately describes the image content'
#             cons_icon  = '✓'
#         elif match_score > 0.45:
#             cons_color = '#f0a500'
#             cons_label = 'PARTIALLY CONSISTENT'
#             cons_desc  = 'Caption partially matches — some elements may be misrepresented'
#             cons_icon  = '◐'
#         else:
#             cons_color = '#ff3b3b'
#             cons_label = 'INCONSISTENT'
#             cons_desc  = 'Caption contradicts image content — strong mismatch signal'
#             cons_icon  = '✗'

#         ca, cb = st.columns([1, 1], gap="large")

#         with ca:
#             # Overall match score
#             st.markdown(f"""
#             <div style="background:var(--bg2);border:1px solid {cons_color}33;
#                         border-radius:8px;padding:1.5rem;text-align:center;
#                         box-shadow:0 0 20px {cons_color}15">
#                 <div style="font-size:2.5rem">{cons_icon}</div>
#                 <div style="font-family:var(--display);font-size:1.5rem;
#                             font-weight:800;color:{cons_color};
#                             letter-spacing:0.1em;margin-top:0.3rem">
#                     {cons_label}
#                 </div>
#                 <div style="font-family:var(--mono);font-size:0.65rem;
#                             color:var(--text-dim);margin-top:0.5rem">
#                     {cons_desc}
#                 </div>
#                 <div style="font-family:var(--mono);font-size:2rem;
#                             font-weight:700;color:{cons_color};margin-top:1rem">
#                     {match_score*100:.1f}%
#                 </div>
#                 <div style="font-family:var(--mono);font-size:0.6rem;
#                             color:var(--text-dim);letter-spacing:0.1em">
#                     CAPTION MATCH SCORE
#                 </div>
#                 <div style="background:var(--border);border-radius:2px;
#                             height:6px;margin-top:0.8rem;overflow:hidden">
#                     <div style="width:{match_score*100:.0f}%;height:100%;
#                                 background:{cons_color};border-radius:2px"></div>
#                 </div>
#             </div>""", unsafe_allow_html=True)

#             # Caption display
#             st.markdown(f"""
#             <div style="background:var(--bg3);border:1px solid var(--border);
#                         border-radius:8px;padding:1rem;margin-top:0.8rem">
#                 <div style="font-family:var(--mono);font-size:0.6rem;
#                             color:var(--text-dim);letter-spacing:0.1em;
#                             margin-bottom:0.5rem">CAPTION PROVIDED</div>
#                 <div style="font-family:var(--sans);font-size:0.88rem;
#                             color:var(--text);font-style:italic;line-height:1.5">
#                     "{caption.strip()}"
#                 </div>
#             </div>""", unsafe_allow_html=True)

#         with cb:
#             # Keyword-level analysis
#             st.markdown(f"""
#             <div style="font-family:var(--mono);font-size:0.6rem;color:var(--text-dim);
#                         letter-spacing:0.1em;margin-bottom:0.8rem">
#                 KEYWORD PRESENCE ANALYSIS
#             </div>""", unsafe_allow_html=True)

#             if word_scores:
#                 for word, score in sorted(word_scores.items(),
#                                           key=lambda x: x[1], reverse=True):
#                     if score > 0.65:
#                         w_color = '#00e676'
#                         w_label = 'FOUND'
#                     elif score > 0.40:
#                         w_color = '#f0a500'
#                         w_label = 'UNCERTAIN'
#                     else:
#                         w_color = '#ff3b3b'
#                         w_label = 'NOT FOUND'

#                     bar_w = int(score * 100)
#                     st.markdown(f"""
#                     <div style="background:var(--bg2);border:1px solid var(--border);
#                                 border-radius:6px;padding:0.7rem 0.9rem;
#                                 margin-bottom:0.5rem">
#                         <div style="display:flex;justify-content:space-between;
#                                     align-items:center;margin-bottom:0.4rem">
#                             <span style="font-family:var(--mono);font-size:0.75rem;
#                                          color:var(--text);text-transform:lowercase">
#                                 "{word}"
#                             </span>
#                             <span style="font-family:var(--mono);font-size:0.65rem;
#                                          color:{w_color}">{score*100:.0f}% · {w_label}</span>
#                         </div>
#                         <div style="background:var(--border);border-radius:2px;
#                                     height:4px;overflow:hidden">
#                             <div style="width:{bar_w}%;height:100%;
#                                         background:{w_color};border-radius:2px"></div>
#                         </div>
#                     </div>""", unsafe_allow_html=True)
#             else:
#                 st.markdown("""
#                 <div style="font-family:var(--mono);font-size:0.7rem;
#                             color:var(--text-dim);padding:1rem">
#                     No keywords extracted — caption too short
#                 </div>""", unsafe_allow_html=True)

#             # Interpretation box — build text OUTSIDE f-string to avoid HTML leakage
#             mismatch_words = [w for w, s in word_scores.items() if s < 0.40]
#             matched_words  = [w for w, s in word_scores.items() if s > 0.65]

#             # Build interpretation sentence cleanly
#             interp_parts = []
#             if matched_words:
#                 joined = ", ".join(matched_words)
#                 interp_parts.append(
#                     f'Keywords <span style="color:#00e676"><b>{joined}</b></span> '
#                     f'appear present in the image.'
#                 )
#             if mismatch_words:
#                 joined = ", ".join(mismatch_words)
#                 interp_parts.append(
#                     f'Keywords <span style="color:#ff3b3b"><b>{joined}</b></span> '
#                     f'were NOT detected — possible caption mismatch.'
#                 )
#             if not mismatch_words and matched_words:
#                 interp_parts.append('Caption appears to accurately describe this image.')
#             if not mismatch_words and not matched_words:
#                 interp_parts.append('No strong keywords found for analysis.')

#             interp_text = ' '.join(interp_parts)

#             st.markdown(
#                 f'<div style="background:var(--bg3);border:1px solid var(--border);'
#                 f'border-left:3px solid {cons_color};border-radius:6px;'
#                 f'padding:1rem;margin-top:0.5rem">'
#                 f'<div style="font-family:var(--mono);font-size:0.6rem;'
#                 f'color:var(--text-dim);letter-spacing:0.1em;margin-bottom:0.5rem">'
#                 f'AI INTERPRETATION</div>'
#                 f'<div style="font-family:var(--sans);font-size:0.8rem;'
#                 f'color:var(--text);line-height:1.7">{interp_text}</div>'
#                 f'</div>',
#                 unsafe_allow_html=True
#             )

#     st.markdown("<br>", unsafe_allow_html=True)

#     # ── PANEL 2: PRESENTATION STATS BAR ─────────────────────
#     st.markdown("---")
#     st.markdown('<div class="sec-label">▸ MODEL PERFORMANCE — PRESENTATION OVERVIEW</div>',
#                 unsafe_allow_html=True)

#     st.markdown("""
#     <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;
#                 padding:1.5rem 2rem;margin-bottom:1rem">
#         <div style="font-family:var(--display);font-size:1rem;font-weight:700;
#                     color:var(--amber);letter-spacing:0.15em;margin-bottom:1.2rem">
#             INTELLIGENT MEDIA FORGERY DETECTION — CSEDS 6TH SEM PROJECT
#         </div>
#         <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:1rem">
#     """, unsafe_allow_html=True)

#     stats = [
#         ("93.1%",  "TEST ACCURACY",     "#00e676"),
#         ("0.9285", "F1 SCORE",          "#00e676"),
#         ("0.9798", "AUC-ROC",           "#00e676"),
#         ("0.9425", "BEST VAL F1",       "#f0a500"),
#         ("94.44%", "BEST VAL ACC",      "#f0a500"),
#         ("14",     "OPTIMAL EPOCH",     "#00b4d8"),
#     ]

#     cols = st.columns(6)
#     for col, (val, label, color) in zip(cols, stats):
#         with col:
#             st.markdown(f"""
#             <div style="background:var(--bg3);border:1px solid var(--border);
#                         border-top:2px solid {color};border-radius:6px;
#                         padding:1rem 0.5rem;text-align:center">
#                 <div style="font-family:var(--mono);font-size:1.4rem;
#                             font-weight:700;color:{color}">{val}</div>
#                 <div style="font-family:var(--mono);font-size:0.55rem;
#                             color:var(--text-dim);margin-top:0.3rem;
#                             letter-spacing:0.08em">{label}</div>
#             </div>""", unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     # Architecture row
#     arch_cols = st.columns(4)
#     arch_items = [
#         ("👁️ VISION BRANCH",    "EfficientNet-B4\n99.8M total params\nImageNet pretrained\nLast 2 blocks fine-tuned"),
#         ("📝 TEXT BRANCH",      "DistilBERT Base\nUncased tokenizer\n128 max token length\nLast transformer block tuned"),
#         ("🔀 FUSION MODULE",    "Cross-Attention\n8 attention heads\n512-dim fusion space\nVision queries text"),
#         ("🗺️ LOCALIZATION",     "U-Net Decoder\n5 upsampling blocks\n1792 → 32 channels\n224×224 heatmap output"),
#     ]
#     for col, (title, desc) in zip(arch_cols, arch_items):
#         with col:
#             lines = desc.split('\n')
#             lines_html = ''.join(
#                 f'<div style="margin-bottom:0.2rem">{l}</div>' for l in lines
#             )
#             st.markdown(f"""
#             <div style="background:var(--bg3);border:1px solid var(--border);
#                         border-radius:6px;padding:1rem">
#                 <div style="font-family:var(--mono);font-size:0.7rem;font-weight:700;
#                             color:var(--amber);margin-bottom:0.6rem">{title}</div>
#                 <div style="font-family:var(--mono);font-size:0.62rem;
#                             color:var(--text-dim);line-height:1.8">
#                     {lines_html}
#                 </div>
#             </div>""", unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     # Training details row
#     st.markdown("""
#     <div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;
#                 padding:1rem 1.5rem;font-family:var(--mono);font-size:0.65rem;
#                 color:var(--text-dim);line-height:2.2">
#         <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2rem">
#             <div>
#                 <div style="color:var(--amber);font-weight:700;margin-bottom:0.3rem">DATASET</div>
#                 Flickr8k · 8,091 images<br>
#                 Train: 70% · Val: 15% · Test: 15%<br>
#                 SBI augmentation on-the-fly<br>
#                 Caption perturbation for fakes
#             </div>
#             <div>
#                 <div style="color:var(--amber);font-weight:700;margin-bottom:0.3rem">TRAINING</div>
#                 Platform: Google Colab T4 GPU<br>
#                 Optimizer: AdamW (lr=2e-4)<br>
#                 Scheduler: CosineAnnealingLR<br>
#                 Loss: Focal + Dice · Batch: 8
#             </div>
#             <div>
#                 <div style="color:var(--amber);font-weight:700;margin-bottom:0.3rem">DETECTION LAYERS</div>
#                 L1: EfficientNet-B4 + DistilBERT<br>
#                 L2: CLIP ViT-B/32 (HuggingFace)<br>
#                 L3: DCT Frequency Analysis<br>
#                 L4: Caption-Image Consistency
#             </div>
#         </div>
#     </div>""", unsafe_allow_html=True)


# # =============================================================
# # FOOTER
# # =============================================================
# st.markdown("""
# <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);
#             font-family:var(--mono);font-size:0.6rem;color:var(--text-dim);
#             display:flex;justify-content:space-between;letter-spacing:0.08em">
#     <span>SENTINEL FORENSICS v3.0</span>
#     <span>L1: EfficientNet-B4 + DistilBERT · L2: CLIP ViT-B/32 · L3: DCT · L4: Caption Analysis</span>
#     <span>F1: 0.9285 · AUC-ROC: 0.9798</span>
# </div>""", unsafe_allow_html=True)

# =============================================================
# app.py — Intelligent Media Forgery Detection (v3)
# Modernized UI: Forensic Lab / Cyberpunk Terminal aesthetic
# Run with: streamlit run app.py
# =============================================================

import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib
from matplotlib import colormaps
from PIL import Image
import sys, os
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
# CSS — FORENSIC LAB / CYBERPUNK TERMINAL
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@700;800&display=swap');

/* ── Base ─────────────────────────────────────────────── */
:root {
    --bg:        #080b10;
    --bg2:       #0d1117;
    --bg3:       #111820;
    --border:    #1e2d3d;
    --amber:     #f0a500;
    --amber-dim: #f0a50044;
    --amber-glow:#f0a50022;
    --red:       #ff3b3b;
    --red-dim:   #ff3b3b33;
    --green:     #00e676;
    --green-dim: #00e67622;
    --cyan:      #00b4d8;
    --text:      #c8d6e5;
    --text-dim:  #5a7a8a;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow', sans-serif;
    --display:   'Barlow Condensed', sans-serif;
}

* { box-sizing: border-box; }

.stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: var(--sans);
}

/* Scanline overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Hide Streamlit chrome ────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border); }

/* ── Header ───────────────────────────────────────────── */
.sentinel-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.5rem 0 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    position: relative;
}
.sentinel-logo {
    width: 52px; height: 52px;
    border: 2px solid var(--amber);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    background: var(--amber-glow);
    box-shadow: 0 0 20px var(--amber-dim);
    flex-shrink: 0;
}
.sentinel-title {
    font-family: var(--display);
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    color: var(--amber);
    text-shadow: 0 0 30px var(--amber-dim);
    line-height: 1;
}
.sentinel-sub {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    margin-top: 0.3rem;
}
.sentinel-badge {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--green);
    background: var(--green-dim);
    border: 1px solid var(--green);
    padding: 0.3rem 0.7rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
}

/* ── Panel base ────────────────────────────────────────── */
.panel {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem;
    position: relative;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
    opacity: 0.6;
}

/* ── Section label ─────────────────────────────────────── */
.sec-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Verdict boxes ─────────────────────────────────────── */
.verdict-box {
    border-radius: 8px;
    padding: 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-authentic {
    background: linear-gradient(135deg, #00e67608, #00e67603);
    border: 1px solid var(--green);
    box-shadow: 0 0 30px #00e67615, inset 0 0 30px #00e67605;
}
.verdict-ai {
    background: linear-gradient(135deg, #f0a50008, #f0a50003);
    border: 1px solid var(--amber);
    box-shadow: 0 0 30px var(--amber-dim), inset 0 0 30px var(--amber-glow);
}
.verdict-fake {
    background: linear-gradient(135deg, #ff3b3b08, #ff3b3b03);
    border: 1px solid var(--red);
    box-shadow: 0 0 30px #ff3b3b22, inset 0 0 30px #ff3b3b08;
}
.verdict-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.verdict-label {
    font-family: var(--display);
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.12em;
}
.verdict-label-auth  { color: var(--green); text-shadow: 0 0 20px #00e67655; }
.verdict-label-ai    { color: var(--amber); text-shadow: 0 0 20px var(--amber-dim); }
.verdict-label-fake  { color: var(--red);   text-shadow: 0 0 20px #ff3b3b55; }
.verdict-desc {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-top: 0.5rem;
    letter-spacing: 0.08em;
}

/* ── Score cards ────────────────────────────────────────── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 1rem;
}
.score-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 0.8rem;
    text-align: center;
}
.score-num {
    font-family: var(--mono);
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1;
}
.score-lbl {
    font-family: var(--mono);
    font-size: 0.58rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
    text-transform: uppercase;
}

/* ── Layer cards ────────────────────────────────────────── */
.layer-row {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    overflow: hidden;
}
.layer-row::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
}
.layer-row-green::before  { background: var(--green); }
.layer-row-amber::before  { background: var(--amber); }
.layer-row-red::before    { background: var(--red);   }
.layer-row-dim::before    { background: var(--border);}

.layer-icon { font-size: 1.3rem; flex-shrink: 0; }
.layer-info { flex: 1; }
.layer-name {
    font-family: var(--sans);
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text);
}
.layer-desc {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-dim);
    margin-top: 0.15rem;
}
.layer-score-wrap { text-align: right; }
.layer-score {
    font-family: var(--mono);
    font-size: 1.3rem;
    font-weight: 700;
}
.layer-risk {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.08em;
}

/* Progress bar */
.bar-bg {
    background: var(--border);
    border-radius: 2px;
    height: 4px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

/* ── Upload zone ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg2) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--amber) !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--amber) !important;
    color: var(--amber) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--amber-glow) !important;
    box-shadow: 0 0 20px var(--amber-dim) !important;
}
.stButton > button:disabled {
    border-color: var(--border) !important;
    color: var(--text-dim) !important;
    opacity: 0.5 !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
    background: var(--amber-glow) !important;
}

/* ── Sidebar ────────────────────────────────────────────── */
.sidebar-title {
    font-family: var(--display);
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    color: var(--amber);
    margin-bottom: 0.3rem;
}
.sidebar-metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.72rem;
}
.sidebar-metric-val { color: var(--amber); font-weight: 700; }

/* ── Info / empty state ─────────────────────────────────── */
.empty-state {
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 3.5rem 2rem;
    text-align: center;
}
.empty-icon { font-size: 2.8rem; opacity: 0.4; }
.empty-text {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 1rem;
    line-height: 1.8;
    letter-spacing: 0.05em;
}

/* ── Expander ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Alerts ─────────────────────────────────────────────── */
.stAlert {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    border-radius: 6px !important;
}

/* ── Slider ─────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    color: var(--amber) !important;
}

/* ── Textarea / text input ──────────────────────────────── */
textarea, input[type="text"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 1px var(--amber-dim) !important;
}

/* ── Metrics (native) ───────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.8rem !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    color: var(--amber) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
}

/* ── Spinner ─────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--amber) !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================
# DETECTION LOGIC (unchanged from v2)
# =============================================================

def dct_frequency_analysis(image_pil):
    img_gray   = np.array(image_pil.convert('L')).astype(np.float32)
    img_gray   = cv2.resize(img_gray, (256, 256))
    dct_coeffs = scipy_dct(scipy_dct(img_gray, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_log    = np.log(np.abs(dct_coeffs) + 1e-8)
    h, w       = dct_log.shape
    hf_energy  = np.mean(np.abs(dct_log[h//2:, w//2:]))
    lf_energy  = np.mean(np.abs(dct_log[:h//4, :w//4]))
    hf_ratio   = hf_energy / (lf_energy + 1e-8)
    dct_abs    = np.abs(dct_coeffs)
    peak_ratio = np.sum(dct_abs > np.percentile(dct_abs, 99)) / dct_abs.size
    noise_floor= np.std(dct_log[h//2:, w//2:])
    hf_score   = max(0, min(1, 1 - (hf_ratio / 0.15)))
    peak_score = max(0, min(1, peak_ratio / 0.005))
    noise_score= max(0, min(1, 1 - (noise_floor / 2.5)))
    ai_score   = 0.45*hf_score + 0.35*peak_score + 0.20*noise_score
    smoothness = 1.0 / (np.std(np.array(image_pil.convert('RGB')).astype(np.float32)) / 50.0 + 1.0)
    ai_score   = min(1.0, ai_score + 0.15 * smoothness)
    return float(ai_score), {'hf_ratio': float(hf_ratio), 'peak_ratio': float(peak_ratio),
                              'noise_floor': float(noise_floor), 'smoothness': float(smoothness)}


@st.cache_resource
def load_clip_model():
    try:
        from transformers import CLIPProcessor, CLIPModel
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        return model, processor, True
    except Exception:
        return None, None, False


def clip_ai_detection(image_pil, clip_model, clip_processor):
    try:
        real_prompts = [
            "a real photograph taken by a camera",
            "a natural photo with realistic lighting and shadows",
            "a genuine photograph of a real scene",
            "a photo taken by a person with a smartphone",
        ]
        ai_prompts = [
            "an AI generated image created by artificial intelligence",
            "a fake image generated by a GAN or diffusion model",
            "an artificially generated synthetic image",
            "an image created by MidJourney or DALL-E or Stable Diffusion",
            "a digital artwork created by a neural network",
        ]
        inputs = clip_processor(text=real_prompts + ai_prompts,
                                images=image_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs   = torch.softmax(outputs.logits_per_image.squeeze(0), dim=0)
        real_score = probs[:len(real_prompts)].sum().item()
        ai_score   = probs[len(real_prompts):].sum().item()
        return float(ai_score / (ai_score + real_score + 1e-8)), True
    except Exception:
        return 0.5, False


@st.cache_resource
def load_sbi_model():
    device    = torch.device('cpu')
    model     = ForgeryDetector()
    weights   = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(weights['model_state'])
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer, device


def run_sbi_inference(image_pil, caption, model, tokenizer, device):
    IMG_SIZE   = 224
    image_np   = cv2.resize(np.array(image_pil.convert('RGB')), (IMG_SIZE, IMG_SIZE))
    transform  = T.Compose([T.ToTensor(),
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img_tensor = transform(image_np).unsqueeze(0).to(device)
    enc        = tokenizer(caption.strip() or "an image", padding='max_length',
                           max_length=128, truncation=True, return_tensors='pt')
    with torch.no_grad():
        logit, heatmap = model(img_tensor, enc['input_ids'].to(device),
                               enc['attention_mask'].to(device))
    return torch.sigmoid(logit).item(), heatmap.squeeze().cpu().numpy(), image_np


def analyze_caption_image_consistency(image_pil, caption, clip_model, clip_processor, clip_ok):
    """
    Accurate caption-image consistency using multi-probe CLIP scoring.

    APPROACH:
    1. MATCH SCORE: Compare caption against 8 diverse alternative descriptions
       (not just 4 fillers). More alternatives = better discrimination.
       We also run the INVERSE: compare image against caption vs antonyms.
       Final score = average of both directions.

    2. KEYWORD SCORES: Each keyword is probed with 4 CLIP templates
       (not just 2), then averaged. More templates = stabler estimate.
       We also calibrate against a neutral baseline so scores are centered.
    """
    if not clip_ok or not caption.strip():
        return None

    try:
        user_caption = caption.strip()

        # ── MATCH SCORE (bidirectional) ───────────────────────────────
        # Direction 1: How much does the image prefer this caption
        # over a diverse set of unrelated descriptions?
        alternatives = [
            user_caption,
            "a photograph of something completely unrelated",
            "an abstract texture or pattern",
            "a blank white or black background",
            "a completely different scene with no connection",
            "random objects with no coherent subject",
            "an outdoor landscape with no people or animals",
            "an indoor scene with furniture",
            "a close-up of food or drink",
        ]
        inp1 = clip_processor(text=alternatives, images=image_pil,
                              return_tensors="pt", padding=True)
        with torch.no_grad():
            out1   = clip_model(**inp1)
            probs1 = torch.softmax(out1.logits_per_image.squeeze(0), dim=0)
            # caption is index 0; random baseline = 1/9 = 0.111
            raw1   = float(probs1[0].item())
            # rescale: 0.111 (random) → 0.0,  1.0 → 1.0
            score1 = max(0.0, min(1.0, (raw1 - 0.111) / (1.0 - 0.111)))

        # Direction 2: Ask CLIP directly — does this image match this caption?
        # Use caption vs a hard negative (explicit contradiction)
        hard_negatives = [
            user_caption,
            f"this image has nothing to do with: {user_caption}",
            f"the opposite of: {user_caption}",
        ]
        inp2 = clip_processor(text=hard_negatives, images=image_pil,
                              return_tensors="pt", padding=True)
        with torch.no_grad():
            out2   = clip_model(**inp2)
            probs2 = torch.softmax(out2.logits_per_image.squeeze(0), dim=0)
            # caption is index 0; random baseline = 1/3 = 0.333
            raw2   = float(probs2[0].item())
            score2 = max(0.0, min(1.0, (raw2 - 0.333) / (1.0 - 0.333)))

        # Final match score = weighted average of both directions
        match_score = 0.55 * score1 + 0.45 * score2

        # ── KEYWORD SCORES (multi-template calibrated) ────────────────
        stopwords = {'a','an','the','is','are','was','were','be','been',
                     'has','have','had','do','does','did','will','would',
                     'can','could','should','may','might','of','in','on',
                     'at','to','for','with','by','from','and','or','but',
                     'not','this','that','it','its','very','just','some'}

        words     = user_caption.lower().split()
        key_words = [w for w in words if len(w) > 2 and w not in stopwords][:6]
        word_scores = {}

        for word in key_words:
            # 4 positive templates + 4 negative templates
            # More templates = more stable CLIP estimate
            pos_templates = [
                f"a photo of {word}",
                f"an image showing {word}",
                f"a picture that contains {word}",
                f"a scene with {word} visible",
            ]
            neg_templates = [
                f"a photo with no {word}",
                f"an image where {word} is absent",
                f"a picture that does not show {word}",
                f"a scene without any {word}",
            ]
            all_templates = pos_templates + neg_templates

            inp_w = clip_processor(text=all_templates, images=image_pil,
                                   return_tensors="pt", padding=True)
            with torch.no_grad():
                out_w  = clip_model(**inp_w)
                pr_w   = torch.softmax(out_w.logits_per_image.squeeze(0), dim=0)

            pos_score = float(pr_w[:4].sum().item())   # sum of 4 positive template probs
            neg_score = float(pr_w[4:].sum().item())   # sum of 4 negative template probs

            # Normalize: pos / (pos + neg) gives calibrated presence probability
            presence = pos_score / (pos_score + neg_score + 1e-8)

            # Calibrate: random baseline = 0.5 (equal pos/neg templates)
            # Rescale so 0.5 (uncertain) stays 0.5 on display
            # Below 0.4 = genuinely absent, above 0.6 = genuinely present
            word_scores[word] = float(presence)

        return {
            'match_score': match_score,
            'word_scores': word_scores,
            'caption'    : user_caption,
            'score1'     : score1,   # for debug display
            'score2'     : score2,
        }
    except Exception:
        return None


def combine_scores(sbi, clip_s, dct, clip_ok, caption_analysis=None):
    """
    Improved score combiner:
    - Uses caption mismatch as an ADDITIONAL signal when caption is provided
    - Applies non-linear boosting: if 2+ layers agree, confidence rises faster
    - Calibrated so real images consistently score below 0.40
    """
    if clip_ok:
        base = 0.35 * sbi + 0.40 * clip_s + 0.25 * dct
    else:
        base = 0.55 * sbi + 0.45 * dct

    # Caption mismatch boost: if caption contradicts image, raise fake score
    if caption_analysis is not None:
        match = caption_analysis['match_score']
        # Low match score = high mismatch = stronger fake signal
        mismatch_signal = max(0.0, 0.5 - match)   # 0 when match≥0.5, up to 0.5 when match=0
        base = min(1.0, base + 0.25 * mismatch_signal)

    # Non-linear agreement boost:
    # If 2 or more layers independently score high, it's much more likely fake
    scores_list = [sbi, dct]
    if clip_ok:
        scores_list.append(clip_s)
    high_count = sum(1 for s in scores_list if s > 0.60)

    if high_count >= 2:
        base = min(1.0, base + 0.08)   # two layers agree → boost
    if high_count == 3:
        base = min(1.0, base + 0.05)   # all three agree → extra boost

    return float(base)


def get_verdict(sbi, clip_s, dct, final, threshold, clip_ok, caption_analysis=None):
    if final < threshold:
        return "AUTHENTIC"
    scores   = {'SBI': sbi, 'DCT': dct}
    if clip_ok:
        scores['CLIP'] = clip_s
    # Caption mismatch is the strongest AI-generation signal
    if caption_analysis and caption_analysis['match_score'] < 0.30:
        scores['CAPTION'] = 1.0 - caption_analysis['match_score']
    dominant = max(scores.keys(), key=lambda k: scores[k])
    if dominant == 'CLIP'    and clip_s > 0.55:  return "AI_GENERATED"
    if dominant == 'CAPTION' and clip_ok:        return "AI_GENERATED"
    if dominant == 'SBI'     and sbi    > 0.55:  return "SPLICED"
    return "MANIPULATED"


def create_overlay(image_np, heatmap_np, alpha=0.45):
    colored = colormaps['jet'](heatmap_np)[:, :, :3]
    overlay = alpha * colored + (1-alpha) * (image_np / 255.0)
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def layer_color_class(s):
    if s > 0.65: return 'layer-row-red',   '#ff3b3b', 'HIGH RISK'
    if s > 0.45: return 'layer-row-amber', '#f0a500', 'ELEVATED'
    return              'layer-row-green', '#00e676', 'CLEAR'


# =============================================================
# LOAD MODELS
# =============================================================
with st.spinner("INITIALIZING SENTINEL SYSTEMS..."):
    sbi_model, tokenizer, device         = load_sbi_model()
    clip_model, clip_processor, clip_ok  = load_clip_model()


# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ SYSTEM STATUS</div>', unsafe_allow_html=True)

    layers = [
        ("L1 SBI SPLICE DETECTOR",  "ONLINE", "#00e676"),
        ("L2 CLIP SEMANTIC ENGINE", "ONLINE" if clip_ok else "OFFLINE",
         "#00e676" if clip_ok else "#ff3b3b"),
        ("L3 DCT FREQ ANALYZER",    "ONLINE", "#00e676"),
    ]
    for name, status, color in layers:
        st.markdown(f"""
        <div class="sidebar-metric">
            <span>{name}</span>
            <span class="sidebar-metric-val" style="color:{color}">● {status}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 MODEL STATS</div>', unsafe_allow_html=True)
    for label, val in [("ACCURACY","93.1%"),("F1 SCORE","0.9285"),("AUC-ROC","0.9798"),
                       ("TRAIN EPOCHS","14 / 15"),("BEST VAL F1","0.9425")]:
        st.markdown(f"""
        <div class="sidebar-metric">
            <span>{label}</span>
            <span class="sidebar-metric-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎚 SENSITIVITY</div>', unsafe_allow_html=True)
    threshold = st.slider("Detection threshold", 0.10, 0.90, 0.45, 0.05,
                          label_visibility="collapsed")
    st.markdown(f"""
    <div style="font-family:var(--mono);font-size:0.65rem;color:var(--text-dim);margin-top:0.3rem">
        THRESHOLD: <span style="color:var(--amber)">{threshold:.2f}</span>
        &nbsp;|&nbsp; {'SENSITIVE' if threshold < 0.4 else 'BALANCED' if threshold < 0.6 else 'CONSERVATIVE'}
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">ℹ ABOUT</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:var(--mono);font-size:0.68rem;color:var(--text-dim);line-height:1.8">
        SENTINEL detects image forgeries<br>
        using a three-layer AI pipeline.<br><br>
        L1 → splice & copy-paste<br>
        L2 → AI/GAN/diffusion origin<br>
        L3 → frequency domain artifacts
    </div>""", unsafe_allow_html=True)


# =============================================================
# HEADER
# =============================================================
st.markdown(f"""
<div class="sentinel-header">
    <div class="sentinel-logo">🛡️</div>
    <div>
        <div class="sentinel-title">SENTINEL</div>
        <div class="sentinel-sub">INTELLIGENT MEDIA FORENSICS SYSTEM — v3.0</div>
    </div>
    <div class="sentinel-badge">
        {'● ALL SYSTEMS ONLINE' if clip_ok else '● L2 OFFLINE — 2/3 LAYERS ACTIVE'}
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# MAIN LAYOUT
# =============================================================
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: INPUT ──────────────────────────────────────────────
with col_left:
    st.markdown('<div class="sec-label">▸ EVIDENCE INPUT</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "DROP IMAGE FILE",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed"
    )
    st.markdown("""
    <div style="font-family:var(--mono);font-size:0.62rem;color:var(--text-dim);
                margin-top:-0.5rem;margin-bottom:0.8rem;letter-spacing:0.05em">
        ACCEPTED: JPG · PNG · WEBP · Max size depends on system memory
    </div>""", unsafe_allow_html=True)

    caption = st.text_area(
        "SUBJECT CAPTION",
        placeholder="Describe image content (optional)...\nEx: a dog running on a beach at sunset",
        height=90,
        label_visibility="collapsed"
    )
    st.markdown("""
    <div style="font-family:var(--mono);font-size:0.62rem;color:var(--text-dim);
                margin-top:-0.5rem;margin-bottom:1rem;letter-spacing:0.05em">
        CAPTION IS OPTIONAL — improves semantic mismatch detection
    </div>""", unsafe_allow_html=True)

    analyze_btn = st.button(
        "▶  INITIATE FORENSIC ANALYSIS",
        use_container_width=True,
        disabled=(uploaded_file is None)
    )

    if uploaded_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">▸ INPUT PREVIEW</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)

        # File metadata
        file_bytes = uploaded_file.getvalue()
        img_check  = Image.open(uploaded_file)
        st.markdown(f"""
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
                    padding:0.8rem 1rem;margin-top:0.5rem;font-family:var(--mono);
                    font-size:0.65rem;color:var(--text-dim);line-height:2">
            FILENAME: <span style="color:var(--text)">{uploaded_file.name}</span><br>
            DIMENSIONS: <span style="color:var(--text)">{img_check.size[0]} × {img_check.size[1]} px</span><br>
            FILE SIZE: <span style="color:var(--text)">{len(file_bytes)/1024:.1f} KB</span><br>
            COLOR MODE: <span style="color:var(--text)">{img_check.mode}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state" style="margin-top:1rem">
            <div class="empty-icon">📁</div>
            <div class="empty-text">
                NO FILE LOADED<br>
                DROP AN IMAGE ABOVE TO BEGIN<br>
                FORENSIC ANALYSIS
            </div>
        </div>""", unsafe_allow_html=True)


# ── RIGHT: RESULTS ────────────────────────────────────────────
with col_right:
    st.markdown('<div class="sec-label">▸ ANALYSIS REPORT</div>', unsafe_allow_html=True)

    if uploaded_file is not None and analyze_btn:
        image_pil = Image.open(uploaded_file)

        # Run all layers
        with st.spinner("[ L1 ] RUNNING SBI SPLICE DETECTION..."):
            sbi_score, heatmap_np, image_np = run_sbi_inference(
                image_pil, caption, sbi_model, tokenizer, device
            )
        with st.spinner("[ L2 ] RUNNING CLIP SEMANTIC ANALYSIS..."):
            clip_score, _ = (clip_ai_detection(image_pil, clip_model, clip_processor)
                             if clip_ok else (0.5, False))
        with st.spinner("[ L3 ] RUNNING DCT FREQUENCY ANALYSIS..."):
            dct_score, dct_details = dct_frequency_analysis(image_pil)

        with st.spinner("[ L4 ] ANALYSING CAPTION-IMAGE CONSISTENCY..."):
            caption_analysis = analyze_caption_image_consistency(
                image_pil, caption, clip_model, clip_processor, clip_ok
            )

        final_score = combine_scores(sbi_score, clip_score, dct_score,
                                     clip_ok, caption_analysis)
        verdict     = get_verdict(sbi_score, clip_score, dct_score,
                                  final_score, threshold, clip_ok, caption_analysis)

        # ── VERDICT ──
        verdict_cfg = {
            "AUTHENTIC"   : ("verdict-authentic", "verdict-label-auth",  "✓", "AUTHENTIC",    "NO MANIPULATION DETECTED"),
            "AI_GENERATED": ("verdict-ai",         "verdict-label-ai",   "⚠", "AI GENERATED", "SYNTHETIC ORIGIN DETECTED"),
            "SPLICED"     : ("verdict-fake",        "verdict-label-fake", "✗", "SPLICED",      "COPY-PASTE FORGERY DETECTED"),
            "MANIPULATED" : ("verdict-fake",        "verdict-label-fake", "✗", "MANIPULATED",  "TAMPERING DETECTED"),
        }
        vbox, vlbl, vicon, vtxt, vdesc = verdict_cfg[verdict]

        st.markdown(f"""
        <div class="verdict-box {vbox}">
            <div class="verdict-icon">{vicon}</div>
            <div class="verdict-label {vlbl}">{vtxt}</div>
            <div class="verdict-desc">{vdesc} · CONFIDENCE: {final_score*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # ── SCORE CARDS ──
        fake_color = '#ff3b3b' if verdict != 'AUTHENTIC' else '#00e676'
        st.markdown(f"""
        <div class="score-grid">
            <div class="score-card">
                <div class="score-num" style="color:{fake_color}">{final_score*100:.1f}<span style="font-size:0.9rem">%</span></div>
                <div class="score-lbl">FAKE SCORE</div>
            </div>
            <div class="score-card">
                <div class="score-num" style="color:#00e676">{(1-final_score)*100:.1f}<span style="font-size:0.9rem">%</span></div>
                <div class="score-lbl">AUTH SCORE</div>
            </div>
            <div class="score-card">
                <div class="score-num" style="color:#f0a500">{threshold*100:.0f}<span style="font-size:0.9rem">%</span></div>
                <div class="score-lbl">THRESHOLD</div>
            </div>
        </div>
        <br>""", unsafe_allow_html=True)

        # ── LAYER BREAKDOWN ──
        st.markdown('<div class="sec-label">▸ LAYER ANALYSIS</div>', unsafe_allow_html=True)

        layers_data = [
            ("🔬", "L1 — SBI SPLICE DETECTOR",   "Copy-paste & region splicing detection", sbi_score,   True),
            ("🧠", "L2 — CLIP SEMANTIC ENGINE",   "GAN · Diffusion · MidJourney · DALL-E",  clip_score,  clip_ok),
            ("📡", "L3 — DCT FREQUENCY ANALYZER", "Unnatural frequency pattern detection",   dct_score,   True),
        ]

        for icon, name, desc, score, active in layers_data:
            if active:
                row_cls, score_col, risk_lbl = layer_color_class(score)
                bar_w = int(score * 100)
                st.markdown(f"""
                <div class="layer-row {row_cls}">
                    <div class="layer-icon">{icon}</div>
                    <div class="layer-info">
                        <div class="layer-name">{name}</div>
                        <div class="layer-desc">{desc}</div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width:{bar_w}%;background:{score_col}"></div>
                        </div>
                    </div>
                    <div class="layer-score-wrap">
                        <div class="layer-score" style="color:{score_col}">{score*100:.1f}%</div>
                        <div class="layer-risk"  style="color:{score_col}">{risk_lbl}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="layer-row layer-row-dim" style="opacity:0.4">
                    <div class="layer-icon">{icon}</div>
                    <div class="layer-info">
                        <div class="layer-name">{name}</div>
                        <div class="layer-desc" style="color:#ff3b3b">OFFLINE — pip install openai-clip</div>
                    </div>
                    <div class="layer-score-wrap">
                        <div class="layer-score" style="color:var(--border)">N/A</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── HEATMAP ──
        st.markdown('<div class="sec-label">▸ SPLICE LOCALIZATION MAP</div>',
                    unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["OVERLAY", "HEATMAP", "ORIGINAL"])
        overlay_np = create_overlay(image_np, heatmap_np, alpha=0.45)
        with tab1:
            st.image(overlay_np, use_container_width=True)
            st.markdown("""<div style="font-family:var(--mono);font-size:0.62rem;
                color:var(--text-dim);margin-top:0.3rem;text-align:center">
                RED/YELLOW REGIONS INDICATE SUSPECTED TAMPERING</div>""",
                unsafe_allow_html=True)
        with tab2:
            hm_rgb = (colormaps['jet'](heatmap_np)[:,:,:3]*255).astype(np.uint8)
            st.image(hm_rgb, use_container_width=True)
        with tab3:
            st.image(image_np, use_container_width=True)

        # ── FORENSIC SUMMARY ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">▸ FORENSIC SUMMARY</div>',
                    unsafe_allow_html=True)

        forged_pct   = float((heatmap_np > 0.5).mean() * 100)
        caption_used = caption.strip() if caption.strip() else "(none)"

        st.markdown(f"""
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
                    padding:1rem 1.2rem;font-family:var(--mono);font-size:0.68rem;
                    line-height:2.2;color:var(--text-dim)">
            VERDICT          : <span style="color:var(--amber);font-weight:700">{verdict}</span><br>
            FINAL FAKE SCORE : <span style="color:var(--text)">{final_score*100:.2f}%</span><br>
            THRESHOLD USED   : <span style="color:var(--text)">{threshold*100:.0f}%</span><br>
            SUSPECTED AREA   : <span style="color:var(--text)">{forged_pct:.1f}% of image</span><br>
            CAPTION INPUT    : <span style="color:var(--text);font-style:italic">{caption_used}</span><br>
            DCT HF RATIO     : <span style="color:var(--text)">{dct_details['hf_ratio']:.4f}</span><br>
            DCT NOISE FLOOR  : <span style="color:var(--text)">{dct_details['noise_floor']:.4f}</span>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🛡️</div>
            <div class="empty-text">
                SENTINEL STANDING BY<br><br>
                UPLOAD AN IMAGE AND CLICK<br>
                [ INITIATE FORENSIC ANALYSIS ]<br><br>
                DETECTS: AI GENERATED · SPLICED<br>
                COPY-PASTE · FREQUENCY ARTIFACTS
            </div>
        </div>""", unsafe_allow_html=True)


# =============================================================
# BOTTOM PANELS — shown after analysis only
# =============================================================
if uploaded_file is not None and analyze_btn:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=False)

    # ── PANEL 1: CAPTION-IMAGE CONSISTENCY ──────────────────
    st.markdown('<div class="sec-label">▸ CAPTION — IMAGE CONSISTENCY ANALYSIS</div>',
                unsafe_allow_html=True)

    if not caption.strip():
        st.markdown("""
        <div style="background:var(--bg2);border:1px dashed var(--border);border-radius:8px;
                    padding:1.5rem;text-align:center;font-family:var(--mono);
                    font-size:0.72rem;color:var(--text-dim)">
            NO CAPTION PROVIDED — Enter a description above and re-analyze
            to see image-caption consistency scoring
        </div>""", unsafe_allow_html=True)

    elif not clip_ok:
        st.markdown("""
        <div style="background:var(--bg2);border:1px dashed var(--border);border-radius:8px;
                    padding:1.5rem;text-align:center;font-family:var(--mono);
                    font-size:0.72rem;color:var(--text-dim)">
            CLIP NOT AVAILABLE — Caption analysis requires CLIP.<br>
            Run: pip install openai-clip
        </div>""", unsafe_allow_html=True)

    elif caption_analysis:
        match_score  = caption_analysis['match_score']
        word_scores  = caption_analysis['word_scores']

        # Consistency verdict
        if match_score > 0.65:
            cons_color = '#00e676'
            cons_label = 'CONSISTENT'
            cons_desc  = 'Caption accurately describes the image content'
            cons_icon  = '✓'
        elif match_score > 0.45:
            cons_color = '#f0a500'
            cons_label = 'PARTIALLY CONSISTENT'
            cons_desc  = 'Caption partially matches — some elements may be misrepresented'
            cons_icon  = '◐'
        else:
            cons_color = '#ff3b3b'
            cons_label = 'INCONSISTENT'
            cons_desc  = 'Caption contradicts image content — strong mismatch signal'
            cons_icon  = '✗'

        ca, cb = st.columns([1, 1], gap="large")

        with ca:
            # Overall match score
            st.markdown(f"""
            <div style="background:var(--bg2);border:1px solid {cons_color}33;
                        border-radius:8px;padding:1.5rem;text-align:center;
                        box-shadow:0 0 20px {cons_color}15">
                <div style="font-size:2.5rem">{cons_icon}</div>
                <div style="font-family:var(--display);font-size:1.5rem;
                            font-weight:800;color:{cons_color};
                            letter-spacing:0.1em;margin-top:0.3rem">
                    {cons_label}
                </div>
                <div style="font-family:var(--mono);font-size:0.65rem;
                            color:var(--text-dim);margin-top:0.5rem">
                    {cons_desc}
                </div>
                <div style="font-family:var(--mono);font-size:2rem;
                            font-weight:700;color:{cons_color};margin-top:1rem">
                    {match_score*100:.1f}%
                </div>
                <div style="font-family:var(--mono);font-size:0.6rem;
                            color:var(--text-dim);letter-spacing:0.1em">
                    CAPTION MATCH SCORE
                </div>
                <div style="background:var(--border);border-radius:2px;
                            height:6px;margin-top:0.8rem;overflow:hidden">
                    <div style="width:{match_score*100:.0f}%;height:100%;
                                background:{cons_color};border-radius:2px"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Caption display
            st.markdown(f"""
            <div style="background:var(--bg3);border:1px solid var(--border);
                        border-radius:8px;padding:1rem;margin-top:0.8rem">
                <div style="font-family:var(--mono);font-size:0.6rem;
                            color:var(--text-dim);letter-spacing:0.1em;
                            margin-bottom:0.5rem">CAPTION PROVIDED</div>
                <div style="font-family:var(--sans);font-size:0.88rem;
                            color:var(--text);font-style:italic;line-height:1.5">
                    "{caption.strip()}"
                </div>
            </div>""", unsafe_allow_html=True)

        with cb:
            # Keyword-level analysis
            st.markdown(f"""
            <div style="font-family:var(--mono);font-size:0.6rem;color:var(--text-dim);
                        letter-spacing:0.1em;margin-bottom:0.8rem">
                KEYWORD PRESENCE ANALYSIS
            </div>""", unsafe_allow_html=True)

            if word_scores:
                for word, score in sorted(word_scores.items(),
                                          key=lambda x: x[1], reverse=True):
                    # Calibrated thresholds:
                    # >0.60 = genuinely present  (was 0.65)
                    # 0.42–0.60 = uncertain
                    # <0.42 = genuinely absent   (was 0.40)
                    if score > 0.60:
                        w_color = '#00e676'
                        w_label = 'FOUND'
                    elif score > 0.42:
                        w_color = '#f0a500'
                        w_label = 'UNCERTAIN'
                    else:
                        w_color = '#ff3b3b'
                        w_label = 'NOT FOUND'

                    bar_w = int(score * 100)
                    st.markdown(f"""
                    <div style="background:var(--bg2);border:1px solid var(--border);
                                border-radius:6px;padding:0.7rem 0.9rem;
                                margin-bottom:0.5rem">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:0.4rem">
                            <span style="font-family:var(--mono);font-size:0.75rem;
                                         color:var(--text);text-transform:lowercase">
                                "{word}"
                            </span>
                            <span style="font-family:var(--mono);font-size:0.65rem;
                                         color:{w_color}">{score*100:.0f}% · {w_label}</span>
                        </div>
                        <div style="background:var(--border);border-radius:2px;
                                    height:4px;overflow:hidden">
                            <div style="width:{bar_w}%;height:100%;
                                        background:{w_color};border-radius:2px"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-family:var(--mono);font-size:0.7rem;
                            color:var(--text-dim);padding:1rem">
                    No keywords extracted — caption too short
                </div>""", unsafe_allow_html=True)

            # Interpretation box — build text OUTSIDE f-string to avoid HTML leakage
            mismatch_words = [w for w, s in word_scores.items() if s < 0.42]
            matched_words  = [w for w, s in word_scores.items() if s > 0.60]

            # Build interpretation sentence cleanly
            interp_parts = []
            if matched_words:
                joined = ", ".join(matched_words)
                interp_parts.append(
                    f'Keywords <span style="color:#00e676"><b>{joined}</b></span> '
                    f'appear present in the image.'
                )
            if mismatch_words:
                joined = ", ".join(mismatch_words)
                interp_parts.append(
                    f'Keywords <span style="color:#ff3b3b"><b>{joined}</b></span> '
                    f'were NOT detected — possible caption mismatch.'
                )
            if not mismatch_words and matched_words:
                interp_parts.append('Caption appears to accurately describe this image.')
            if not mismatch_words and not matched_words:
                interp_parts.append('No strong keywords found for analysis.')

            interp_text = ' '.join(interp_parts)

            st.markdown(
                f'<div style="background:var(--bg3);border:1px solid var(--border);'
                f'border-left:3px solid {cons_color};border-radius:6px;'
                f'padding:1rem;margin-top:0.5rem">'
                f'<div style="font-family:var(--mono);font-size:0.6rem;'
                f'color:var(--text-dim);letter-spacing:0.1em;margin-bottom:0.5rem">'
                f'AI INTERPRETATION</div>'
                f'<div style="font-family:var(--sans);font-size:0.8rem;'
                f'color:var(--text);line-height:1.7">{interp_text}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PANEL 2: PRESENTATION STATS BAR ─────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-label">▸ MODEL PERFORMANCE — PRESENTATION OVERVIEW</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;
                padding:1.5rem 2rem;margin-bottom:1rem">
        <div style="font-family:var(--display);font-size:1rem;font-weight:700;
                    color:var(--amber);letter-spacing:0.15em;margin-bottom:1.2rem">
            INTELLIGENT MEDIA FORGERY DETECTION — CSEDS 6TH SEM PROJECT
        </div>
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:1rem">
    """, unsafe_allow_html=True)

    stats = [
        ("93.1%",  "TEST ACCURACY",     "#00e676"),
        ("0.9285", "F1 SCORE",          "#00e676"),
        ("0.9798", "AUC-ROC",           "#00e676"),
        ("0.9425", "BEST VAL F1",       "#f0a500"),
        ("94.44%", "BEST VAL ACC",      "#f0a500"),
        ("14",     "OPTIMAL EPOCH",     "#00b4d8"),
    ]

    cols = st.columns(6)
    for col, (val, label, color) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div style="background:var(--bg3);border:1px solid var(--border);
                        border-top:2px solid {color};border-radius:6px;
                        padding:1rem 0.5rem;text-align:center">
                <div style="font-family:var(--mono);font-size:1.4rem;
                            font-weight:700;color:{color}">{val}</div>
                <div style="font-family:var(--mono);font-size:0.55rem;
                            color:var(--text-dim);margin-top:0.3rem;
                            letter-spacing:0.08em">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture row
    arch_cols = st.columns(4)
    arch_items = [
        ("👁️ VISION BRANCH",    "EfficientNet-B4\n99.8M total params\nImageNet pretrained\nLast 2 blocks fine-tuned"),
        ("📝 TEXT BRANCH",      "DistilBERT Base\nUncased tokenizer\n128 max token length\nLast transformer block tuned"),
        ("🔀 FUSION MODULE",    "Cross-Attention\n8 attention heads\n512-dim fusion space\nVision queries text"),
        ("🗺️ LOCALIZATION",     "U-Net Decoder\n5 upsampling blocks\n1792 → 32 channels\n224×224 heatmap output"),
    ]
    for col, (title, desc) in zip(arch_cols, arch_items):
        with col:
            lines = desc.split('\n')
            lines_html = ''.join(
                f'<div style="margin-bottom:0.2rem">{l}</div>' for l in lines
            )
            st.markdown(f"""
            <div style="background:var(--bg3);border:1px solid var(--border);
                        border-radius:6px;padding:1rem">
                <div style="font-family:var(--mono);font-size:0.7rem;font-weight:700;
                            color:var(--amber);margin-bottom:0.6rem">{title}</div>
                <div style="font-family:var(--mono);font-size:0.62rem;
                            color:var(--text-dim);line-height:1.8">
                    {lines_html}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training details row
    st.markdown("""
    <div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;
                padding:1rem 1.5rem;font-family:var(--mono);font-size:0.65rem;
                color:var(--text-dim);line-height:2.2">
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2rem">
            <div>
                <div style="color:var(--amber);font-weight:700;margin-bottom:0.3rem">DATASET</div>
                Flickr8k · 8,091 images<br>
                Train: 70% · Val: 15% · Test: 15%<br>
                SBI augmentation on-the-fly<br>
                Caption perturbation for fakes
            </div>
            <div>
                <div style="color:var(--amber);font-weight:700;margin-bottom:0.3rem">TRAINING</div>
                Platform: Google Colab T4 GPU<br>
                Optimizer: AdamW (lr=2e-4)<br>
                Scheduler: CosineAnnealingLR<br>
                Loss: Focal + Dice · Batch: 8
            </div>
            <div>
                <div style="color:var(--amber);font-weight:700;margin-bottom:0.3rem">DETECTION LAYERS</div>
                L1: EfficientNet-B4 + DistilBERT<br>
                L2: CLIP ViT-B/32 (HuggingFace)<br>
                L3: DCT Frequency Analysis<br>
                L4: Caption-Image Consistency
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# =============================================================
# FOOTER
# =============================================================
st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);
            font-family:var(--mono);font-size:0.6rem;color:var(--text-dim);
            display:flex;justify-content:space-between;letter-spacing:0.08em">
    <span>SENTINEL FORENSICS v3.0</span>
    <span>L1: EfficientNet-B4 + DistilBERT · L2: CLIP ViT-B/32 · L3: DCT · L4: Caption Analysis</span>
    <span>F1: 0.9285 · AUC-ROC: 0.9798</span>
</div>""", unsafe_allow_html=True)