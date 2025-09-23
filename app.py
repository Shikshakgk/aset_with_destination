import streamlit as st
import cv2, numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, zipfile, tempfile, shutil

st.title("💎 ASET Batch Analyzer with Destination Path")

# Upload multiple files
uploaded_files = st.file_uploader(
    "📂 Upload ASET images (multiple files allowed)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Ask for destination folder path
destination = st.text_input("📍 Enter Destination Folder Path (must exist on server/local machine):")

if uploaded_files and destination:
    if st.button("▶ Process and Save"):
        if not os.path.exists(destination):
            st.error("❌ Destination folder does not exist. Please create it first.")
        else:
            with st.spinner("Processing images..."):
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    basename = os.path.splitext(filename)[0]
                    result_folder = os.path.join(destination, basename)
                    os.makedirs(result_folder, exist_ok=True)

                    # Read image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # ---- your processing steps here (same as before) ----
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    kernel = np.ones((5, 5), np.uint8)
                    diamond_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    coords = cv2.findNonZero(diamond_mask)
                    x, y, w, h = cv2.boundingRect(coords)
                    img_cropped = img_rgb[y:y + h, x:x + w]
                    diamond_mask_cropped = diamond_mask[y:y + h, x:x + w]
                    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)

                    # Red / Green / Blue / Grey masks (same logic)
                    red1 = cv2.inRange(hsv, (0, 90, 50), (10, 255, 255))
                    red2 = cv2.inRange(hsv, (170, 90, 50), (179, 255, 255))
                    red_mask = cv2.bitwise_or(red1, red2)
                    green_mask = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
                    blue_mask = cv2.inRange(hsv, (90, 60, 40), (140, 255, 255))
                    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
                    gray_mask  = cv2.inRange(hsv, (0, 0, 80), (180, 50, 200))
                    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))

                    diamond_area = np.count_nonzero(diamond_mask_cropped)
                    red_count   = np.count_nonzero(cv2.bitwise_and(red_mask, diamond_mask_cropped))
                    green_count = np.count_nonzero(cv2.bitwise_and(green_mask, diamond_mask_cropped))
                    blue_count  = np.count_nonzero(cv2.bitwise_and(blue_mask, diamond_mask_cropped))
                    black_count = np.count_nonzero(cv2.bitwise_and(black_mask, diamond_mask_cropped))
                    gray_count  = np.count_nonzero(cv2.bitwise_and(gray_mask, diamond_mask_cropped))
                    white_count = np.count_nonzero(cv2.bitwise_and(white_mask, diamond_mask_cropped))
                    grey_count = black_count + gray_count + white_count

                    percentages = {
                        "Red":   100 * red_count / diamond_area,
                        "Green": 100 * green_count / diamond_area,
                        "Blue":  100 * blue_count / diamond_area,
                        "Others": 100 * grey_count / diamond_area
                    }

                    overlay = np.zeros_like(img_cropped)
                    overlay[red_mask > 0]   = [255, 0, 0]
                    overlay[green_mask > 0] = [0, 255, 0]
                    overlay[blue_mask > 0]  = [0, 0, 255]
                    overlay[black_mask > 0] = [128, 128, 128]
                    overlay[gray_mask > 0]  = [128, 128, 128]
                    overlay[white_mask > 0] = [128, 128, 128]
                    blended = cv2.addWeighted(img_cropped, 0.6, overlay, 0.4, 0)

                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    axes[0, 0].imshow(img_rgb); axes[0, 0].set_title("Original"); axes[0, 0].axis('off')
                    axes[0, 1].imshow(img_cropped); axes[0, 1].set_title("Cropped"); axes[0, 1].axis('off')
                    axes[1, 0].imshow(blended); axes[1, 0].set_title("Detected Colors"); axes[1, 0].axis('off')
                    axes[1, 1].pie(percentages.values(), labels=percentages.keys(),
                                autopct='%1.1f%%', colors=['#FF0000', '#00FF00', '#0000FF', '#808080'])
                    axes[1, 1].set_title("Color Distribution")
                    plt.tight_layout()

                    fig.savefig(os.path.join(result_folder, f"{basename}_analysis.png"))
                    plt.close(fig)

                    df = pd.DataFrame(list(percentages.items()), columns=["Color", "Percentage"])
                    df.to_csv(os.path.join(result_folder, f"{basename}_results.csv"), index=False)

            st.success(f"✅ Processing complete! Results saved in: {destination}")
