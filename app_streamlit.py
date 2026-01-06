import streamlit as st
from PIL import Image, PngImagePlugin
from io import BytesIO
import torch
import torchvision.transforms as T
import base64
import numpy as np

from src.model.risranet import RISRANet
from src.config import DEVICE


def pil_to_tensor(img):
    tf = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return tf(img).unsqueeze(0)

def tensor_to_pil(t):
    t = t.detach().cpu().clamp(-1,1)
    img = ((t + 1) / 2).squeeze(0)
    return T.ToPILImage()(img)

def tensor_to_base64(t):
    arr = t.detach().cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode("utf-8")

def base64_to_tensor(b64, shape, device):
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape)
    return torch.tensor(arr, device=device)

def pil_to_png_bytes(pil_img, pnginfo=None):
    buf = BytesIO()
    if pnginfo:
        pil_img.save(buf, format="PNG", pnginfo=pnginfo)
    else:
        pil_img.save(buf, format="PNG")
    return buf.getvalue()


# Load model (cached)

@st.cache_resource
def load_model(ckpt_path="checkpoints/best.pt"):
    model = RISRANet(in_channels=3).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

MODEL = load_model()


# Page config + CSS

st.set_page_config(page_title="Steganography Lab", layout="wide")

st.markdown(
    r"""
    <style>
    /* soft pastel background */
    [data-testid="stAppViewContainer"] {
      background: radial-gradient(circle at 30% 30%, #f8fbfd 0%, #eef7fb 40%, #f3f9ff 100%);
      color: #12313a;
    }

    /* compact header block (smaller padding so no big blank area) */
    .header-block {
      backdrop-filter: blur(8px) saturate(120%);
      background-color: rgba(255,255,255,0.85);
      border-radius: 12px;
      padding: 14px 24px;
      margin: 12px auto 8px auto;
      border: 1px solid rgba(170,190,200,0.12);
      max-width: 1200px;
      box-shadow: 0 6px 18px rgba(40,60,70,0.05);
    }

    h1 {
      text-align:center !important;
      color:#15343a !important;
      font-size: 40px !important;
      margin: 0;
      font-weight: 800;
    }
    h4 {
      text-align:center !important;
      color:#4a6a70 !important;
      margin: 6px 0 0 0;
      font-weight: 600;
    }

    /* make tabs larger and bolder */
    .stTabs [data-baseweb="tab"] {
        font-size: 30px;
        font-weight: 800;
        padding-top: 14px;
        padding-bottom: 14px;
        color: #12343b !important;
    }

    /* main content container */
    .block-container {
      backdrop-filter: blur(8px) saturate(120%);
      background-color: rgba(255,255,255,0.9);
      border-radius: 12px;
      padding: 18px;
      margin-top: 12px;
      border: 1px solid rgba(180,200,210,0.10);
      box-shadow: 0 6px 20px rgba(40,60,70,0.04);
      max-width: 1200px;
      margin-left: auto;
      margin-right: auto;
    }

    /* Upload box styling */
    .upload-box {
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,251,252,0.95));
      padding: 16px;
      border-radius: 12px;
      border: 1px solid rgba(120,150,160,0.06);
      box-shadow: 0 6px 16px rgba(40,60,70,0.03);
      min-height: 120px;
    }
    .uploader-title {
      font-weight:700; color:#15343a; margin-bottom:8px; text-align:center;
    }

    /* ensure Browse files button visible */
    .stFileUploader button {
      background: #f3fbff !important;
      color: #15343a !important;
      border-radius: 8px;
      padding: 6px 10px;
      border: 1px solid rgba(30,70,80,0.06);
    }

    /* image preview container */
    .soft-frame {
      border-radius: 12px;
      padding: 8px;
      background: rgba(255,255,255,0.88);
      border: 1px solid rgba(50,80,90,0.06);
      width: 68%;
      margin-left: auto;
      margin-right: auto;
      margin-bottom: 16px;
    }
    img { max-height: 420px; object-fit: contain; display:block; margin-left:auto; margin-right:auto; }

    /* buttons visible, high contrast */
    .stButton > button {
      background: linear-gradient(90deg,#ffffff,#e6f7ff);
      color: #15343a !important;
      border-radius: 10px;
      padding: 12px 26px;
      font-size: 16px;
      font-weight: 700;
      border: 1px solid rgba(80,120,140,0.18);
      box-shadow: 0 5px 14px rgba(80,120,140,0.08);
      transition: all 0.18s ease-in-out;
    }
    .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 26px rgba(80,120,140,0.12);
    }

    /* tip text */
    .subtle { text-align:center; color:#45636c; font-size:14px; }

    /* layout tweaks to ensure content aligns under tabs tightly */
    .stApp .main > div.block-container:first-of-type { margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Header (compact)
st.markdown("<div class='header-block'>", unsafe_allow_html=True)
st.title("Steganography Lab")
st.markdown("<h4>Hide and reveal images using a reversible neural network</h4>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)



# Tabs & Main content

tab_hide, tab_reveal = st.tabs(["HIDE IMAGE", "REVEAL IMAGE"])

# HIDE TAB 
with tab_hide:
    st.markdown("<div class='block-container'>", unsafe_allow_html=True)
    st.subheader("Upload a cover image and a secret image")

    col1, col2 = st.columns(2)

    # Cover uploader
    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.markdown("<div class='uploader-title'>Cover Image</div>", unsafe_allow_html=True)
        cover_file = st.file_uploader(" ", type=["png","jpg","jpeg"], key="cover_upload")
        st.markdown("</div>", unsafe_allow_html=True)
        if cover_file:
            cover_pil = Image.open(cover_file).convert("RGB")
            st.markdown("<div class='soft-frame'>", unsafe_allow_html=True)
            st.image(cover_pil, caption="Cover", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Secret uploader
    with col2:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.markdown("<div class='uploader-title'>Secret Image</div>", unsafe_allow_html=True)
        secret_file = st.file_uploader(" ", type=["png","jpg","jpeg"], key="secret_upload")
        st.markdown("</div>", unsafe_allow_html=True)
        if secret_file:
            secret_pil = Image.open(secret_file).convert("RGB")
            if cover_file:
                secret_pil = secret_pil.resize(cover_pil.size)
            st.markdown("<div class='soft-frame'>", unsafe_allow_html=True)
            st.image(secret_pil, caption="Secret (resized)", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<p class='subtle'>Tip: For best results use images with similar dimensions.</p>", unsafe_allow_html=True)

    if cover_file and secret_file:
        if st.button("Create stego image"):
            with st.spinner("Hiding secret..."):
                with torch.no_grad():
                    cov_t = pil_to_tensor(cover_pil).to(DEVICE)
                    sec_t = pil_to_tensor(secret_pil).to(DEVICE)
                    stego_t, g = MODEL.hide(cov_t, sec_t)
                    stego_pil = tensor_to_pil(stego_t)

                # embed latent g inside PNG metadata
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text("latent_g", tensor_to_base64(g))
                pnginfo.add_text("latent_shape", str(list(g.shape)))

                stego_bytes = pil_to_png_bytes(stego_pil, pnginfo)

                st.success("Stego image created!")
                st.markdown("<div class='soft-frame'>", unsafe_allow_html=True)
                st.image(stego_pil, caption="Stego Image", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div style='text-align:center; margin-top:10px;'>", unsafe_allow_html=True)
                st.download_button("Download stego (PNG)", stego_bytes,
                                   file_name="stego.png",
                                   mime="image/png")
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- REVEAL TAB ----------
with tab_reveal:
    st.markdown("<div class='block-container'>", unsafe_allow_html=True)
    st.subheader("Upload a stego image")

    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.markdown("<div class='uploader-title'>Stego Image (PNG)</div>", unsafe_allow_html=True)
    stego_file = st.file_uploader(" ", type=["png"], key="stego_upload")
    st.markdown("</div>", unsafe_allow_html=True)

    if stego_file:
        stego_pil = Image.open(stego_file).convert("RGB")
        st.markdown("<div class='soft-frame'>", unsafe_allow_html=True)
        st.image(stego_pil, caption="Stego", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if stego_file and st.button("Reveal secret"):
        png = Image.open(stego_file)
        meta = png.info

        if "latent_g" not in meta:
            st.error("This PNG does not contain the required latent metadata.")
        else:
            g_b64 = meta["latent_g"]
            shape = eval(meta["latent_shape"])
            g = base64_to_tensor(g_b64, shape, DEVICE)

            with st.spinner("Recovering secret..."):
                with torch.no_grad():
                    stego_t = pil_to_tensor(stego_pil).to(DEVICE)
                    _, secret_rec = MODEL.recover(stego_t, g)
                    sec_pil = tensor_to_pil(secret_rec)

            st.success("Secret recovered!")
            st.markdown("<div class='soft-frame'>", unsafe_allow_html=True)
            st.image(sec_pil, caption="Recovered Secret", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='text-align:center; margin-top:10px;'>", unsafe_allow_html=True)
            st.download_button("Download", pil_to_png_bytes(sec_pil),
                               file_name="recovered_secret.png", mime="image/png")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
