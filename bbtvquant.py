import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from scipy.stats import skew, kurtosis

st.set_page_config(layout="wide")  # Make the app use the full browser width
st.title("Leaf Greenness Quantification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to HSV color space for better segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Adjust HSV range for green color (relaxed thresholds)
    lower_green = np.array([30, 40, 20])  # Lower bound of green in HSV
    upper_green = np.array([90, 255, 255])  # Upper bound of green in HSV

    # Create a mask for green areas
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)  # Kernel size can be adjusted
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel)  # Expand leaf areas

    # Apply the refined mask to the original image
    leaf_only_image = cv2.bitwise_and(image, image, mask=green_mask)

    # Remove black background (pixels with low intensity values)
    mask = np.any(leaf_only_image > 10, axis=2)  # Create a mask to exclude black pixels
    image_no_bg = leaf_only_image.copy()
    image_no_bg[~mask] = [0, 0, 0]  # Set black background pixels to [0, 0, 0]

    # Set max display width for canvas
    max_canvas_width = 800
    scale = 1.0
    if image.shape[1] > max_canvas_width:
        scale = max_canvas_width / image.shape[1]
        display_width = max_canvas_width
        display_height = int(image.shape[0] * scale)
        display_image = cv2.resize(image_no_bg, (display_width, display_height))
    else:
        display_width = image.shape[1]
        display_height = image.shape[0]
        display_image = image_no_bg.copy()

    st.write("### Draw rectangles to select your samples (ROIs)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=Image.fromarray(display_image),
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="rect",
        key="canvas",
    )

    # Compute skewed-distribution parameters for each ROI
    results = []
    if canvas_result.json_data is not None:
        for idx, obj in enumerate(canvas_result.json_data["objects"]):
            left = int(obj["left"] / scale)
            top = int(obj["top"] / scale)
            width = int(obj["width"] / scale)
            height = int(obj["height"] / scale)
            x1, y1 = left, top
            x2, y2 = left + width, top + height
            roi_img = image_no_bg[y1:y2, x1:x2]
            if roi_img.size == 0:
                skewness = np.nan
                kurtosis_val = np.nan
                r_mean = np.nan
                r_skewness = np.nan
            else:
                # Mask out black pixels within the ROI
                roi_mask = np.any(roi_img > 10, axis=2)
                R, G, B = cv2.split(roi_img)
                r_values = R[roi_mask].flatten()
                r_mean = np.mean(r_values)
                r_skewness = skew(r_values)
                skewness = skew(G[roi_mask].flatten())
                kurtosis_val = kurtosis(G[roi_mask].flatten())
            
            # Calculate SPAD using the F4 model
            if not np.isnan(r_mean) and not np.isnan(r_skewness):
                spad = (
                    0.3344
                    + 0.8709 * r_mean
                    - 177.3 * r_skewness
                    - 0.005536 * (r_mean ** 2)
                    + 2.876 * r_mean * r_skewness
                    + 8.515 * (r_skewness ** 2)
                    - 0.01227 * (r_mean ** 2) * r_skewness
                    - 0.1398 * r_mean * (r_skewness ** 2)
                    + 7.301 * (r_skewness ** 3)
                )
            else:
                spad = np.nan

            results.append({
                "Sample": idx + 1,
                "Skewness": skewness if roi_img.size != 0 else np.nan,
                "Kurtosis": kurtosis_val if roi_img.size != 0 else np.nan,
                "RMean": r_mean if roi_img.size != 0 else np.nan,
                "RSkewness": r_skewness if roi_img.size != 0 else np.nan,
                "SPAD": spad if roi_img.size != 0 else np.nan,
            })

    if results:
        df = pd.DataFrame(results)
        st.write("### Chlorosis Analysis")
        # Updated formula for chlorosis score
        df["Chlorosis Score"] = abs(df["Skewness"]) + df["Kurtosis"]
        st.dataframe(df)

        # Allow users to download the updated results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "chlorosis_results.csv", "text/csv")