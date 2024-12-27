import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from VisionTransformer import VITdetector, create_vit_object_detector

# Determine the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture and parameters
def load_model():
    st.text("Loading model... Please wait!")
    # Define model parameters
    image_size = 224
    patch_size = 32
    input_shape = (image_size, image_size, 3)
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 4
    mlp_head_units = [2048, 1024, 512, 64, 32]

    # Create model and load weights
    model_layer_list = create_vit_object_detector(
        input_shape, patch_size, num_patches, projection_dim, num_heads,
        transformer_units, transformer_layers, mlp_head_units
    )
    model = VITdetector(model_layer_list)
    model.load_state_dict(torch.load("vit_object_detector.pth", map_location=device))
    model.to(device)
    model.eval()
    st.success("Model loaded successfully!")
    return model

# Load the model
model = load_model()

# Streamlit app title
st.title("🛫 Vision Transformer Airplane Detector")
st.markdown("""
Welcome to the **Vision Transformer (ViT)** app! This tool uses a state-of-the-art Transformer-based model to detect airplanes in uploaded images. 
Follow the steps below to get started:
1. Upload an image.
2. Let the model process it.
3. View the results with a bounding box around detected airplanes!
""")

# Upload image
st.subheader("Step 1: Upload Your Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Processing your image...")

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).squeeze().cpu().numpy()  # Move result to CPU

        # Convert normalized coordinates back to image dimensions
        width, height = image.size
        pred_box = [
            prediction[0] * width,
            prediction[1] * height,
            prediction[2] * width,
            prediction[3] * height,
        ]

        # Draw bounding box on image
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = patches.Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        st.pyplot(fig)

        st.success("Detection complete! Check the bounding box in the displayed image.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
else:
    st.info("Upload an image to start the detection process.")
