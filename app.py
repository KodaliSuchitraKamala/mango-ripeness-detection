import streamlit as st
import requests
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance
import io
import tempfile

# Page configuration
st.set_page_config(
    page_title="Ripen Right: AI-Scanned Mango Freshness",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .instructions {
        background-color: #e9f7ef;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000/predict"  # Update this in production

# Helper functions
def preprocess_image(image):
    """Preprocess the image for display."""
    image = np.array(image)
    # Convert from BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def enhance_image(image):
    """Enhance image for better visualization."""
    # Convert to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Enhance contrast and sharpness
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    
    return image

def predict_image(image):
    """Send image to the API for prediction."""
    try:
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        
        # Send request to API
        files = {"file": ("mango.jpg", buffered.getvalue(), "image/jpeg")}
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from API: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Main application
def main():
    # Header section
    st.markdown("""
    <div class="header">
        <h1>🥭 Ripen Right</h1>
        <h3>AI-Scanned Mango Freshness Detection</h3>
        <p>Upload an image of a mango to detect its ripeness level</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Upload section
        st.markdown("### 📤 Upload Mango Image")
        st.markdown("""
        <div class="instructions">
            <p>• Take a clear photo of a single mango</p>
            <p>• Ensure good lighting and minimal background</p>
            <p>• Center the mango in the frame for best results</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        # Display the uploaded image
        if uploaded_file is not None:
            try:
                # Read and display the image
                image = Image.open(uploaded_file)
                st.image(
                    image,
                    caption="Uploaded Mango",
                    use_column_width=True,
                    output_format="JPEG"
                )
                
                # Make prediction when button is clicked
                if st.button("🔍 Analyze Ripeness", use_container_width=True):
                    with st.spinner("Analyzing mango ripeness..."):
                        # Make prediction
                        result = predict_image(image)
                        
                        if result:
                            # Store results in session state
                            st.session_state.prediction_result = result
                            st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        # Results section
        st.markdown("### 📊 Analysis Results")
        
        if 'prediction_result' in st.session_state and st.session_state.prediction_result:
            result = st.session_state.prediction_result
            top_pred = result['top_prediction']
            
            # Display top prediction with confidence
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            
            # Display result with appropriate color
            if top_pred['class_name'] == 'ripe':
                st.success(f"🍍 **Ripeness Level:** {top_pred['class_name'].title()}")
                st.balloons()
            elif top_pred['class_name'] == 'unRipe':
                st.warning(f"🥭 **Ripeness Level:** {top_pred['class_name'].title()}")
            else:  # overRipe
                st.error(f"🍌 **Ripeness Level:** {top_pred['class_name'].title()}")
            
            # Display confidence
            confidence = top_pred['confidence'] * 100
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.1f}%")
            
            # Display all predictions
            st.markdown("### 📈 Detailed Analysis")
            for pred in result['predictions']:
                confidence = pred['confidence'] * 100
                st.markdown(
                    f"**{pred['class_name'].title()}:** "
                    f"{confidence:.1f}%"
                )
                st.progress(int(confidence))
            
            # Display recommendations based on ripeness
            st.markdown("### 📝 Recommendations")
            if top_pred['class_name'] == 'unRipe':
                st.info("""
                - This mango is not yet ripe
                - Store at room temperature for 2-3 days
                - Check daily for ripeness
                - Ripe when slightly soft to the touch and fragrant
                """)
            elif top_pred['class_name'] == 'ripe':
                st.success("""
                - This mango is perfectly ripe!
                - Best to consume within 1-2 days
                - Store in the refrigerator to slow further ripening
                - Enjoy fresh or use in recipes
                """)
            else:  # overRipe
                st.warning("""
                - This mango is overripe
                - Best used immediately in smoothies or baking
                - Check for any signs of spoilage
                - Consider freezing for later use
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a button to analyze another mango
            if st.button("🔄 Analyze Another Mango", use_container_width=True):
                del st.session_state.prediction_result
                st.experimental_rerun()
        else:
            # Show placeholder when no prediction is made yet
            st.markdown("""
            <div style="text-align: center; padding: 4rem 1rem;">
                <div style="font-size: 5rem;">🥭</div>
                <h3>Upload a mango image to begin analysis</h3>
                <p>Your results will appear here</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add some space at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Ripen Right: AI-Scanned Mango Freshness</p>
        <p>Using deep learning to help you pick the perfect mango</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    # Initialize session state for prediction results
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    main()
