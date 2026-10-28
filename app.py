import streamlit as st
import time
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# MODULE IMPORTS
# ---------------------------------------------------------
from core.tensor_operations import Tensor
from layers.image_convolution import apply_convolution_py, apply_convolution_np
from layers.max_pooling import apply_max_pooling_py, apply_max_pooling_np
from layers.relu_activation import apply_relu_py, apply_relu_np

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Neural-Optic Core",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS FOR ADVANCED UI
# ---------------------------------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Hide the Streamlit Top Bar and Deploy Button */
    header {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;} /* Hides default footer */

    /* 2. Fixed Header Container */
    .fixed-header {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        background-color: #0E1117 !important; /* Default Dark Theme */
        z-index: 999999 !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    @media (prefers-color-scheme: light) {
        .fixed-header {
            background-color: #ffffff !important;
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }
        .sub-title {
            color: #31333F !important;
        }
    }

    /* Shift the main content down so it doesn't hide behind the fixed header */
    .block-container {
        padding-top: 9rem !important; 
    }

    /* Glowing Gradient Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
    .sub-title {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.2rem;
        margin-bottom: 0rem;
        font-weight: 500;
        margin-left: 50px;
    }
    
    /* 5. Custom Footer */
    .custom-footer {
        position: fixed !important;
        left: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        background-color: #262730 !important;
        color: #FAFAFA !important;
        text-align: center;
        padding: 10px;
        font-size: 0.9rem;
        z-index: 999999 !important;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
    }

    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .section-header {
        font-size: 1.75rem;
        font-weight: 600;
        margin-bottom: 1rem;
        margin-top: 1rem;
        margin-left: -5px;
        color: var(--text-color);
    }

    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem !important;
            padding-top: 0.5rem;
        }
        .sub-title {
            font-size: 0.9rem !important;
            margin-left: 10px !important;
        }
        .fixed-header {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        .block-container {
            padding-top: 5rem !important; 
        }
        .section-header {
            font-size: 1.3rem !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
        }
        [data-testid="stMetricValue"] > div {
            font-size: 1.2rem !important;
        }
        
        [data-testid="stMetricLabel"] > div > div {
            font-size: 0.8rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# CORE ENGINE FUNCTIONS
# ---------------------------------------------------------
def load_and_preprocess_image(image_file, engine_mode):
    img = Image.open(image_file).convert('RGB')
    
    if "Custom" in engine_mode:
        img.thumbnail((720, 720))
    
    display_img = np.array(img)
    tensor_img = np.transpose(display_img, (2, 0, 1))
    
    return display_img, tensor_img

def normalize_feature_map(feature_map_array):
    if len(feature_map_array.shape) == 3 and feature_map_array.shape[0] == 1:
        feature_map_array = feature_map_array[0]
        
    f_min, f_max = feature_map_array.min(), feature_map_array.max()
    if f_max - f_min == 0:
        return np.zeros_like(feature_map_array, dtype=np.uint8)
        
    normalized = (feature_map_array - f_min) / (f_max - f_min) * 255
    return normalized.astype(np.uint8)

# ---------------------------------------------------------
# MAIN APP LAYOUT
# ---------------------------------------------------------
def main():
    apply_custom_css()
    
    # 1. Advanced Kernel Dictionary (2D matrices)
    AVAILABLE_KERNELS = {
        "Edge Detection (Vertical)": [
            [1, 0, -1], 
            [1, 0, -1], 
            [1, 0, -1]
        ],
        "Edge Detection (Horizontal)": [
            [1, 1, 1], 
            [0, 0, 0], 
            [-1, -1, -1]
        ],
        "Sharpen": [
            [0, -1, 0], 
            [-1, 5, -1], 
            [0, -1, 0]
        ],
        "Box Blur (Smoothing)": [
            [1, 1, 1], 
            [1, 1, 1], 
            [1, 1, 1]
        ],
        "Emboss (3D Effect)": [
            [-2, -1, 0], 
            [-1, 1, 1], 
            [0, 1, 2]
        ],
        "Ridge Detection": [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]
    }

    st.markdown("""
        <div class="fixed-header">
            <div class="main-title">🧠 Neural-Optic Core</div>
            <div class="sub-title">N-Dimensional Tensor Processing Engine</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">⚙️ Engine Controls</div>', unsafe_allow_html=True)
        
        # Kernel Selection Dropdown
        selected_kernel_name = st.selectbox(
            "Select Convolution Filter:", 
            list(AVAILABLE_KERNELS.keys()),
            help="Choose the mathematical matrix that will slide over the image to extract features."
        )
        
        # 2. Dynamic 3D Kernel Generation
        base_2d_kernel = AVAILABLE_KERNELS[selected_kernel_name]
        # Copy the 2D kernel 3 times to create a 3D kernel (Red, Green, Blue channels)
        kernel_3d = [base_2d_kernel, base_2d_kernel, base_2d_kernel]
        kernel_array = np.array(kernel_3d)
        
        engine_mode = st.radio(
            "Select Processing Backend:",
            ["Custom Pure Python (Architect Mode)", "NumPy Engine (Turbo Mode)"]
        )
        
        uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        # Update the UI text to reflect the new capabilities
        if "Custom" in engine_mode:
            st.warning("⚠️ Architect Mode: Image scaled to max 720px for clear output. It may take a few seconds.")
        else:
            st.success("🚀 Turbo Mode: No Limits. Processing at full original resolution.")
            
        start_btn = st.button("Initialize Processing Pipeline", use_container_width=True, type="primary")

    with col2:
        st.markdown('<div class="section-header">👁️ Optic Visualizer</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Pass the selected engine_mode to the function
            display_img, tensor_img = load_and_preprocess_image(uploaded_file, engine_mode)
            
            # Extract dynamic shape for the UI
            dynamic_h = display_img.shape[0]
            dynamic_w = display_img.shape[1]
            
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(display_img, caption=f"Original Input ({dynamic_h}x{dynamic_w})", use_container_width=True)
            
            if start_btn:
                with st.spinner(f"Executing {selected_kernel_name} Filter..."):
                    start_time = time.time()
                    
                    try:
                        if "Custom" in engine_mode:
                            image_tensor = Tensor(tensor_img.tolist())
                            kernel_tensor = Tensor(kernel_3d)
                            
                            conv_out = apply_convolution_py(image_tensor, kernel_tensor)
                            relu_out = apply_relu_py(conv_out)
                            pool_out = apply_max_pooling_py(relu_out, pool_size=2)
                            
                            final_output_array = np.array(pool_out.data)
                        else:
                            conv_out = apply_convolution_np(tensor_img, kernel_array)
                            relu_out = apply_relu_np(conv_out)
                            pool_out = apply_max_pooling_np(relu_out, pool_size=2)
                            
                            final_output_array = pool_out
                            
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        visual_map = normalize_feature_map(final_output_array)
                        
                        with img_col2:
                            st.image(visual_map, caption=f"Feature Map: {selected_kernel_name}", use_container_width=True, clamp=True)
                            
                        st.markdown("---")
                        st.markdown('<div class="section-header">📊 Pipeline Telemetry</div>', unsafe_allow_html=True)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Execution Time", f"{processing_time:.4f} sec")
                        m2.metric("Input Shape", f"(3, {dynamic_h}, {dynamic_w})")
                        m3.metric("Output Shape", f"{final_output_array.shape}")
                        
                    except Exception as e:
                        st.error(f"Pipeline crashed! Error: {e}")
        else:
            st.info("Awaiting image sequence... Please upload an image from the control panel.")

    st.markdown(
        """
        <div class="custom-footer">
            Engineered with ❤️ by Lokendra Kushwaha | © 2026 The Neural-Optic Core Project
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()