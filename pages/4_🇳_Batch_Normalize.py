import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

def main():
    st.set_page_config(page_title="PyTorch BatchNorm2D Explorer", layout="wide")
    st.title("🔍 Image BatchNorm2D Explorer with PyTorch")
    st.write("""
        Upload an image, configure BatchNorm2D parameters, and visualize the resulting feature maps along with detailed normalization information.
    """)

    # Sidebar for BatchNorm2D parameters
    st.sidebar.header("🔧 BatchNorm2D Parameters")

    # Number of features (input channels)
    num_features = st.sidebar.slider("🎨 Number of Input Channels", min_value=1, max_value=64, value=3, step=1)

    # Batch normalization epsilon
    eps = st.sidebar.slider("✏️ Epsilon", min_value=1e-5, max_value=1e-2, value=1e-5, step=1e-5, format="%.5f")
    
    # Batch normalization momentum
    momentum = st.sidebar.slider("💨 Momentum", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    # Image upload
    uploaded_file = st.file_uploader("📁 Choose an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption='📷 Uploaded Image.', use_column_width=True)
            
            # Display original image size
            st.write(f"**📏 Original Image Size:** {image.size[0]} x {image.size[1]} pixels")
            
            # Image Mode Selection
            mode = st.selectbox("🎨 Select Image Mode for BatchNorm", options=["RGB", "Grayscale"], index=0)
            if mode == "RGB" and image.mode != 'RGB':
                image = image.convert('RGB')
                st.write("✅ Converted image to RGB.")
            elif mode == "Grayscale" and image.mode != 'L':
                image = image.convert('L')
                st.write("✅ Converted image to Grayscale.")
            
            # Convert image to numpy array and normalize
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # If grayscale, add channel dimension
            if mode == "Grayscale":
                image_array = np.expand_dims(image_array, axis=0)  # C x H x W
            else:
                image_array = image_array.transpose(2, 0, 1)  # C x H x W
            
            # Convert to torch tensor and add batch dimension
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # N x C x H x W

            # Display image tensor shape
            st.write(f"**🧮 Image Tensor Shape:** {image_tensor.shape}")
            st.write("**📈 Tensor Details:**")
            st.markdown(f"- **Batch Size:** {image_tensor.shape[0]}")
            st.markdown(f"- **Channels:** {image_tensor.shape[1]}")
            st.markdown(f"- **Height:** {image_tensor.shape[2]}")
            st.markdown(f"- **Width:** {image_tensor.shape[3]}")
            st.markdown("---")

            # Button to apply BatchNorm2D
            if st.button("🚀 Apply BatchNorm2D"):
                with st.spinner('🔄 Applying BatchNorm2D...'):
                    # Define BatchNorm2d layer
                    batch_norm_layer = nn.BatchNorm2d(
                        num_features=num_features,
                        eps=eps,
                        momentum=momentum
                    )

                    # Apply BatchNorm2D
                    batch_norm_layer.eval()  # Set to evaluation mode
                    with torch.no_grad():
                        output = batch_norm_layer(image_tensor)

                    # Move to CPU and convert to numpy
                    output_np = output.squeeze(0).cpu().numpy()  # C_out x H_out x W_out

                    # Display normalization parameters and output tensor shape
                    st.markdown("### 📊 BatchNorm2D Details")
                    st.markdown(f"- **Number of Features (Input Channels):** {num_features}")
                    st.markdown(f"- **Epsilon:** {eps}")
                    st.markdown(f"- **Momentum:** {momentum}")
                    st.markdown(f"- **Final Tensor Shape:** [Batch Size: {output.shape[0]}, Channels: {output.shape[1]}, Height: {output.shape[2]}, Width: {output.shape[3]}]")
                    st.markdown("---")
                    
                    # Display normalized feature maps
                    st.header("📊 Normalized Feature Maps")
                    
                    # Determine number of columns and rows for plotting
                    max_cols = 4
                    num_filters_to_display = output.shape[1]
                    num_cols = min(max_cols, num_filters_to_display)
                    num_rows = math.ceil(num_filters_to_display / num_cols)
                    
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3))
                    
                    # Flatten axes array for easy iteration
                    if num_filters_to_display == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()
                    
                    for i in range(num_filters_to_display):
                        feature_map = output_np[i, :, :]  # Shape: H_out x W_out
                        cmap = 'jet'
                        
                        # Normalize feature map for display
                        feature_map_norm = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-5)
                        
                        axes[i].imshow(feature_map_norm, cmap=cmap)
                        axes[i].axis('off')
                        axes[i].set_title(f'Channel {i+1}')
                    
                    # Hide any unused subplots
                    for j in range(num_filters_to_display, len(axes)):
                        axes[j].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.success("✅ BatchNorm2D applied successfully!")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.write("Please check the console for more details.")

    else:
        st.info("💡 Please upload an image to get started.")

if __name__ == "__main__":
    main()
