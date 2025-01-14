import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from gradcam1 import GradCAM
import numpy as np
from lime import lime_image
import os
import torch.nn.functional as F

# Load the base DenseNet121 model and modify the classifier for binary classification
densenet_model = models.densenet121(pretrained=False)
num_features = densenet_model.classifier.in_features
densenet_model.classifier = nn.Sequential(
    nn.Linear(num_features, 2)  # Binary classification (2 classes: Pneumonia vs. Normal)
)
# Load the saved model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
densenet_model.load_state_dict(torch.load('densenet_model.pth', map_location=device))
densenet_model = densenet_model.to(device)

#  Set the model to evaluation mode
densenet_model.eval()

densenet_model = densenet_model.to('cpu')

# Target layer for Grad-CAM visualization
model_densenet121_layers = densenet_model.features[-1]


 # Plot the results (Original Image, Heatmap, Grad-CAM Result)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot the original and LIME-highlighted images side by side
fig2, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Load an initial image when the app starts
default_image = "person14_bacteria_51.jpeg"


# FUNCTIONS 
###############################################################
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if isinstance(img_path, str):
        # If it's a file path, open the image
        img_path = Image.open(img_path).convert("RGB")

    img_np = np.array(img_path)
    
    if len(img_np.shape) == 2:  # Grayscale image (height, width)
        img_np = np.expand_dims(img_np, axis=-1)  # Add channel dimension (height, width, 1)
        img_np = np.repeat(img_np, 3, axis=-1)    # Convert to 3 channels (height, width, 3)
     # Convert back to PIL image (if needed)
    img_path = Image.fromarray(img_np)
    return transform(img_path).unsqueeze(0)  # Add batch dimension

def preprocess_image2(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the model's input size
        transforms.ToTensor(),         # Convert the image to a tensor
        transforms.Normalize(          # Normalize using ImageNet's mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure the image has 3 channels (RGB)
    return transform(image).unsqueeze(0)  

# Function to predict using the model (wrapper for LIME)
def predict_fn(images):
    images = torch.Tensor(images)  # Ensure data is in the right format
    images = images.permute(0, 3, 1, 2)  # Convert from (batch, height, width, channels) to (batch, channels, height, width)
    
    with torch.no_grad():
        outputs = densenet_model(images)
    
    probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.detach().cpu().numpy()
###############################################################

 
with st.sidebar:

    st.header("Input Image")

    # File uploader for image files (e.g., .jpg, .png)
    uploaded_file  = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:

   
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        # img_path = "person20_bacteria_69.jpeg"

        # Apply Grad-CAM
        gradcam = GradCAM(densenet_model, model_densenet121_layers, image)
        img, heatmap, result, index = gradcam()

       

        # Display original image
        ax[0].imshow(img.permute(1, 2, 0).cpu().numpy())
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        # Display heatmap
        ax[1].imshow(heatmap.permute(1, 2, 0).cpu().numpy(), cmap="jet")
        ax[1].set_title("Heatmap")
        ax[1].axis("off")

        # Display overlayed result
        ax[2].imshow(result.permute(1, 2, 0).cpu().numpy())
        ax[2].set_title("Grad-CAM Result")
        ax[2].axis("off")

        plt.tight_layout()
      
        # FUNCTIONS FOR LIME 
        ###############################################################
        # Preprocess the image for LIME explanation
        input_image = preprocess_image(image)
        input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()

        # LIME expects (batch_size, height, width, channels)
        input_image_np = np.expand_dims(input_image_np, axis=0)

        # Create a LIME Image Explainer
        explainer = lime_image.LimeImageExplainer()

        # Explain the instance
        explanation = explainer.explain_instance(input_image_np[0], predict_fn, top_labels=1, num_samples=100)

        # Get the explanation image and mask
        explanation_img, mask = explanation.get_image_and_mask(1, positive_only=True, num_features=10, hide_rest=False)

        # Ensure the mask is binary
        mask = mask.astype(bool)

        # Create a red mask for the highlighted areas
        red_mask = np.zeros_like(explanation_img)
        red_mask[mask] = [1, 0, 0]  # Set the mask areas to red (RGB: [1, 0, 0])

        # Normalize explanation_img to [0, 1]
        explanation_img = np.clip(explanation_img, 0, 1)

        # Blend the original image with the red mask
        highlighted_image = explanation_img * 0.7 + red_mask * 0.3  # Adjust blending strength

      

        # Display the original image
        axes[0].imshow(explanation_img)
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        # Display the LIME-explained image
        axes[1].imshow(highlighted_image)
        axes[1].axis('off')
        axes[1].set_title('LIME-Highlighted Image')

        plt.tight_layout()

        ###############################################################

        

        # Get model outputs (logits)
        with torch.no_grad():
            logits = densenet_model(input_image)

        # Convert logits to probabilities using softmax
        probabilities = F.softmax(logits, dim=1)

        # Extract the confidence level for Pneumonia (class index 1)
        pneumonia_prob = probabilities[0, 1].item()  # Probability for Pneumonia class
        normal_prob = probabilities[0, 0].item() 

        # Print the results
        st.write(f"Pneumonia Probability: {pneumonia_prob * 100:.2f}%")
        st.write(f"Normal Probability: {normal_prob * 100:.2f}%")

    else:
        # Display the default image before upload
        st.image(default_image, caption="person14_bacteria_51.jpeg", use_container_width=True)

        # Apply Grad-CAM to the default image
        gradcam = GradCAM(densenet_model, model_densenet121_layers, default_image)
        img, heatmap, result, index = gradcam()

         # Display original image
        ax[0].imshow(img.permute(1, 2, 0).cpu().numpy())
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        # Display heatmap
        ax[1].imshow(heatmap.permute(1, 2, 0).cpu().numpy(), cmap="jet")
        ax[1].set_title("Heatmap")
        ax[1].axis("off")

        # Display overlayed result
        ax[2].imshow(result.permute(1, 2, 0).cpu().numpy())
        ax[2].set_title("Grad-CAM Result")
        ax[2].axis("off")

        plt.tight_layout()

        # FUNCTIONS FOR LIME 
        ###############################################################
        # Preprocess the image for LIME explanation
        input_image = preprocess_image(default_image)
        input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()

        # LIME expects (batch_size, height, width, channels)
        input_image_np = np.expand_dims(input_image_np, axis=0)

        # Create a LIME Image Explainer
        explainer = lime_image.LimeImageExplainer()

        # Explain the instance
        explanation = explainer.explain_instance(input_image_np[0], predict_fn, top_labels=1, num_samples=100)

        # Get the explanation image and mask
        explanation_img, mask = explanation.get_image_and_mask(1, positive_only=True, num_features=10, hide_rest=False)

        # Ensure the mask is binary
        mask = mask.astype(bool)

        # Create a red mask for the highlighted areas
        red_mask = np.zeros_like(explanation_img)
        red_mask[mask] = [1, 0, 0]  # Set the mask areas to red (RGB: [1, 0, 0])

        # Normalize explanation_img to [0, 1]
        explanation_img = np.clip(explanation_img, 0, 1)

        # Blend the original image with the red mask
        highlighted_image = explanation_img * 0.7 + red_mask * 0.3  # Adjust blending strength

      
        # Display the original image
        axes[0].imshow(explanation_img)
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        # Display the LIME-explained image
        axes[1].imshow(highlighted_image)
        axes[1].axis('off')
        axes[1].set_title('LIME-Highlighted Image')

        plt.tight_layout()

        ###############################################################

        input_image = preprocess_image2(default_image)

        # Get model outputs (logits)
        with torch.no_grad():
            logits = densenet_model(input_image)

        # Convert logits to probabilities using softmax
        probabilities = F.softmax(logits, dim=1)

        # Extract the confidence level for Pneumonia (class index 1)
        pneumonia_prob = probabilities[0, 1].item()  # Probability for Pneumonia class
        normal_prob = probabilities[0, 0].item() 

        # Print the results
        st.write(f"Pneumonia Probability: {pneumonia_prob * 100:.2f}%")
        st.write(f"Normal Probability: {normal_prob * 100:.2f}%")


 # Add tabs to the Streamlit app
# tab1, tab2 = st.tabs(["Dashboard", "Statistics"])
       
# with tab1:
st.write(" ## DASHBOARD")

# Step 1: Track the filename before st.sidebar
if uploaded_file is not None:
    image_filename = uploaded_file.name
else:
    image_filename = default_image  # Use the default image filename

# Display the filename in the main section before the sidebar
st.markdown("<h5>Pneumonia Image Explanation with Grad-CAM & LIME on DenseNet121</h5>", unsafe_allow_html=True)
# st.write(f"### Pneumonia Image Explanation on DenseNet121")
st.write(f"Results for Image: {image_filename})")



# For Grad-CAM section
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Grad-CAM</h3>", unsafe_allow_html=True)
st.pyplot(fig)
# Grad-CAM explanation
st.write("**Grad-CAM:** Grad-CAM (Gradient-weighted Class Activation Mapping) highlights regions in the X-ray image that contribute most to the model's classification, using a heatmap with warm colors (red, yellow) to emphasize critical areas for prediction.")


# For LIME section
st.markdown("<h3 style='text-align: center; color: #FF6347;'>LIME</h3>", unsafe_allow_html=True)
st.pyplot(fig2)
# LIME explanation
st.write("**LIME:** LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the model with a simpler, interpretable model, highlighting important features using a transparent overlay. Red color was used to indicate the most important areas contributing to the prediction.")
# with tab2:
#     st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Statistics</h2>", unsafe_allow_html=True)
    
#     # Add some statistics about the model
#     st.write("### Model Statistics")
#     st.write("- Model: DenseNet121")
#     st.write("- Classes: Normal, Pneumonia")
#     st.write("- Total Parameters: {:,}".format(sum(p.numel() for p in densenet_model.parameters())))

