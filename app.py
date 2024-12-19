import streamlit as st
from fastai.learner import load_learner
from PIL import Image
import torch
from pathlib import Path
import io

# Set page config
st.set_page_config(
    page_title="🌭 Hotdog Classifier",
    page_icon="🌭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better text readability and styling
st.markdown("""
    <style>
        /* Make the sidebar thinner */
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 200px;
            max-width: 200px;
        }
        
        /* Improve text readability */
        .stMarkdown {
            font-size: 1.1rem;
            line-height: 1.6;
        }
        
        /* Style the title */
        h1 {
            color: #1E1E1E;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
        }
        
        /* Style the headers */
        h3 {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-top: 1rem !important;
        }
        
        /* Style the sidebar text */
        .css-1d391kg {
            font-size: 1rem;
        }
        
        /* Style the upload button */
        .stButton>button {
            font-size: 1.1rem;
            font-weight: 500;
            padding: 0.5rem 1rem;
        }
        
        /* Style the file uploader */
        .stFileUploader {
            margin-bottom: 1.5rem;
        }
        
        /* Add some spacing between elements */
        .stImage {
            margin: 2rem 0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)"""
    model_path = Path('hotdog_model.pkl')
    if not model_path.exists():
        st.error("Model file not found! Please make sure 'hotdog_model.pkl' exists in the current directory.")
        st.stop()
    return load_learner(model_path)

def predict_image(img):
    """Make prediction on an image"""
    learn = load_model()
    pred, pred_idx, probs = learn.predict(img)
    return pred, probs[pred_idx]

# Title and description
st.title("🌭 Not Hotdog?")
st.markdown("""
    <div style='font-size: 1.2rem; margin-bottom: 2rem;'>
        Upload an image to check if it's a hotdog! 
        <br>
        <span style='font-size: 0.9rem; color: #666;'>
            Supported formats: JPG, JPEG, PNG
        </span>
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Is it a hotdog? 🔍'):
        with st.spinner('Analyzing...'):
            prediction, confidence = predict_image(image)
            
            # Display result
            is_hotdog = prediction == 'hotdog'
            result_color = '#28a745' if is_hotdog else '#dc3545'  # Green or Red
            
            st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='margin-top: 0;'>Result:</h3>
                    <div style='font-size: 1.4rem; color: {result_color}; font-weight: 600;'>
                        {prediction.upper()}
                    </div>
                    <div style='font-size: 1.1rem; margin-top: 0.5rem;'>
                        Confidence: {confidence:.2%}
                    </div>
                    <div style='font-size: 0.9rem; color: #666; margin-top: 0.5rem;'>
                        {
                            "Very confident!" if confidence > 0.9
                            else "Pretty sure!" if confidence > 0.7
                            else "Not entirely sure..." if confidence > 0.5
                            else "This one's tricky..."
                        }
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Fun emoji response
            if is_hotdog:
                st.balloons()
                st.markdown("""
                    <div style='font-size: 1.3rem; text-align: center; margin: 1rem 0;'>
                        🌭 Yes, it's a hotdog! 🌭
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='font-size: 1.3rem; text-align: center; margin: 1rem 0;'>
                        ❌ Nope, not a hotdog!
                    </div>
                """, unsafe_allow_html=True)

# Add information about the model
st.sidebar.header("About")
st.sidebar.markdown("""
    <div style='font-size: 1rem; line-height: 1.5;'>
        This app uses a FastAI model trained to classify whether an image contains a hotdog or not.
        <br><br>
        Inspired by the 'Silicon Valley' TV show! 🎬
    </div>
""", unsafe_allow_html=True)