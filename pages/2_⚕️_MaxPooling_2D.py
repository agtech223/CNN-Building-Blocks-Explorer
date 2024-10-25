import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

def calculate_output_size(input_size, kernel_size, stride, padding, dilation=1):
    """
    Calculate the size of the output after a pooling operation.
    Formula:
    out = floor((input + 2*padding - dilation*(kernel_size-1) -1)/stride + 1)
    """
    return math.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)

def main():
    st.set_page_config(page_title="PyTorch MaxPool2D Explorer", layout="wide")
    st.title("🔍 Image MaxPool2D Explorer with PyTorch")
    st.write("""
        Upload an image, configure MaxPool2D parameters, and visualize the resulting feature maps along with detailed pooling information.
    """)

    # Sidebar for MaxPool2D parameters
    st.sidebar.header("🔧 MaxPool2D Parameters")

    # Pooling kernel size
    kernel_size = st.sidebar.slider("📐 Pooling Kernel Size", min_value=1, max_value=10, value=2, step=1)
    kernel_size_tuple = (kernel_size, kernel_size)
    
    # Strides
    stride = st.sidebar.slider("↕️ Stride", min_value=1, max_value=4, value=2, step=1)
    stride_tuple = (stride, stride)
    
    # Padding
    padding_option = st.sidebar.selectbox("🛡️ Padding", options=["valid", "same"], index=0)
    padding = kernel_size // 2 if padding_option == "same" else 0

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
            mode = st.selectbox("🎨 Select Image Mode for Pooling", options=["RGB", "Grayscale"], index=0)
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

            # Button to apply MaxPool2D
            if st.button("🚀 Apply MaxPool2D"):
                with st.spinner('🔄 Applying MaxPool2D...'):
                    # Define MaxPool2D layer
                    maxpool_layer = nn.MaxPool2d(kernel_size=kernel_size_tuple, stride=stride_tuple, padding=padding)
                    
                    # Apply MaxPool2D
                    maxpool_layer.eval()  # Set to evaluation mode
                    with torch.no_grad():
                        output = maxpool_layer(image_tensor)
                    
                    # Move to CPU and convert to numpy
                    output_np = output.squeeze(0).cpu().numpy()  # C x H_out x W_out
                    
                    # Calculate output dimensions
                    input_height = image_tensor.shape[2]
                    input_width = image_tensor.shape[3]
                    out_height = calculate_output_size(input_height, kernel_size, stride, padding)
                    out_width = calculate_output_size(input_width, kernel_size, stride, padding)
                    
                    # Display pooling parameters and output tensor shape
                    st.markdown("### 📊 Pooling Details")
                    st.markdown(f"- **Pooling Kernel Size:** {kernel_size_tuple}")
                    st.markdown(f"- **Stride:** {stride_tuple}")
                    st.markdown(f"- **Padding:** {padding}")
                    st.markdown(f"- **Output Feature Map Size:** {out_width} x {out_height} pixels")
                    st.markdown(f"- **Final Tensor Shape:** [Batch Size: {output.shape[0]}, Channels: {output.shape[1]}, Height: {output.shape[2]}, Width: {output.shape[3]}]")
                    st.markdown("---")
                    
                    # Display feature maps
                    st.header("📊 Pooled Feature Maps")
                    
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
                    
                    st.success("✅ MaxPool2D applied successfully!")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.write("Please check the console for more details.")

    else:
        st.info("💡 Please upload an image to get started.")

if __name__ == "__main__":
        main()
