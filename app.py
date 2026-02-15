import numpy as np
from PIL import Image

import streamlit as st
import matplotlib.cm as cm

from stego.detector import StegoDetector
from stego.features import get_device


def overlay_heatmap_on_image(
    img: Image.Image, heatmap, alpha: float = 0.5
) -> Image.Image:
    """
    Overlays Grad-CAM heatmap (H,W) on original image (resized).
    """
    img = img.convert("RGB")
    heatmap_np = heatmap.numpy()
    heatmap_np = np.clip(heatmap_np, 0.0, 1.0)

    img_np = np.array(
        img.resize(heatmap_np.shape[::-1], Image.BILINEAR), dtype=np.float32
    )
    img_np = img_np / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_color = colormap(heatmap_np)[..., :3]

    overlay = alpha * heatmap_color + (1 - alpha) * img_np
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def main():
    st.set_page_config(page_title="Steganography Detector", layout="wide")

    st.title("Steganography Detection Prototype (SRNet + Residuals)")
    st.write(
        "Upload an image (JPEG/PNG/BMP). The system analyzes **noise residuals**, "
        "JPEG **DCT artifacts**, and **ELA** to estimate a **Threat Score** and "
        "visualize likely regions using **Grad-CAM**."
    )

    st.sidebar.header("Model Configuration")
    model_weights_path = st.sidebar.text_input(
        "Optional model weights (.pth)", value=""
    )
    image_size = st.sidebar.slider(
        "Model input size", min_value=128, max_value=512, value=256, step=64
    )
    st.sidebar.write(f"Device: **{get_device()}**")

    uploaded_file = st.file_uploader(
        "Drag and drop an image file", type=["jpg", "jpeg", "png", "bmp"]
    )

    if (
        "detector" not in st.session_state
        or (model_weights_path and st.session_state.get("loaded_weights") != model_weights_path)
        or st.session_state.get("image_size") != image_size
    ):
        with st.spinner("Initializing steganography detector..."):
            detector = StegoDetector(
                image_size=image_size,
                model_weights=model_weights_path if model_weights_path else None,
                device=get_device(),
            )
            st.session_state["detector"] = detector
            st.session_state["loaded_weights"] = model_weights_path
            st.session_state["image_size"] = image_size

    detector: StegoDetector = st.session_state["detector"]

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Failed to open image: {e}")
            return

        col1, col2, col3 = st.columns(3)

        with st.spinner("Running steganalysis..."):
            threat_score, heatmap, ela_img = detector.predict(img)
            overlay = overlay_heatmap_on_image(img, heatmap, alpha=0.5)

        with col1:
            st.subheader("Original Image")
            st.image(img, use_column_width=True)
            st.text(f"Format: {img.format or 'Unknown'}")
            st.text(f"Size: {img.size[0]}x{img.size[1]}")

        with col2:
            st.subheader("Grad-CAM Heatmap")
            st.image(overlay, use_column_width=True)

        with col3:
            st.subheader("Error Level Analysis (ELA)")
            st.image(ela_img, use_column_width=True)

        st.markdown("---")
        st.subheader("Threat Assessment")

        if threat_score >= 80:
            level = "High"
            color = "red"
        elif threat_score >= 50:
            level = "Medium"
            color = "orange"
        else:
            level = "Low"
            color = "green"

        st.markdown(
            f"**Threat Score:** <span style='color:{color}; font-size:24px;'>{threat_score:.2f}% ({level})</span>",
            unsafe_allow_html=True,
        )

        st.write(
            "This prototype score is based on a residual CNN (SRNet-like) trained on noise residual features. "
            "To achieve robust 95%+ accuracy on real-world social media images, fine-tune on **ALASKA2** "
            "and augment with social-media-style recompression."
        )
    else:
        st.info("Upload an image to start steganography analysis.")


if __name__ == "__main__":
    main()

