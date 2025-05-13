import os
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

# Configuration
MODEL_PATH = "model/pneumonia_detection_Vision_Model.keras"  # or "xray_model.h5" depending on which you have
CLASSES = ["Normal", "Abnormal"]  # Update with your actual class names

def load_model():
    """Load the trained model."""
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the image to make it compatible with the model."""
    # Convert to RGB if grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to match model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize to [0,1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_xray(image):
    """Process the image and make a prediction."""
    if image is None:
        return {"Error": "No image provided"}
    
    try:
        # Load model if not already loaded
        global model
        if 'model' not in globals() or model is None:
            model = load_model()
            if model is None:
                return {"Error": "Failed to load model"}
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        # Get class prediction
        class_index = 1 if prediction > 0.5 else 0
        confidence = prediction if class_index == 1 else 1 - prediction
        
        # Format results
        result = {
            "Class": CLASSES[class_index],
            "Confidence": f"{confidence * 100:.2f}%"
        }
        
        return result
    
    except Exception as e:
        return {"Error": f"An error occurred: {str(e)}"}

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="X-Ray Classification System") as app:
        gr.Markdown("# X-Ray Classification System")
        gr.Markdown("Upload an X-ray image for classification.")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload X-ray Image")
                submit_btn = gr.Button("Analyze X-ray", variant="primary")
            
            with gr.Column(scale=1):
                output = gr.JSON(label="Classification Results")
                
        submit_btn.click(
            fn=predict_xray,
            inputs=input_image,
            outputs=output
        )
        
        gr.Markdown("## How to use")
        gr.Markdown("""
        1. Upload an X-ray image using the panel on the left
        2. Click the 'Analyze X-ray' button
        3. View the classification results on the right
        """)
        
        gr.Markdown("## About")
        gr.Markdown("""
        This application uses a deep learning model based on MobileNetV2 
        architecture to classify X-ray images. The model was trained on 
        medical X-ray datasets to identify abnormalities.
        """)
    
    return app

# For standalone execution
if __name__ == "__main__":
    # Make sure the model is loaded before starting the interface
    model = load_model()
    
    # Create and launch the interface
    app = create_interface()
    app.launch(share=True, server_name="127.1.1.1", server_port=7070)