import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

def calculate_output_size_conv2dtranspose(input_size, kernel_size, stride, padding, output_padding, dilation=1):
    """
    Calculate the size of the output after a Conv2DTranspose operation.
    Formula:
    out = (input - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    """
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

def main():
    st.set_page_config(page_title="PyTorch Conv2DTranspose Explorer", layout="wide")
    st.title("🔍 Image Conv2DTranspose (Deconvolution) Explorer with PyTorch")
    st.write("""
        Upload an image, configure Conv2DTranspose parameters, and visualize the resulting feature maps along with detailed information.
    """)

    # Sidebar for Conv2DTranspose parameters
    st.sidebar.header("🔧 Conv2DTranspose Parameters")

    # Kernel size input
    kernel_size = st.sidebar.slider("📐 Kernel Size", min_value=1, max_value=11, value=3, step=1)
    kernel_size_tuple = (kernel_size, kernel_size)
    
    # Number of filters
    num_filters = st.sidebar.slider("🎨 Number of Filters (Output Channels)", min_value=1, max_value=64, value=8, step=1)

    # Strides
    stride = st.sidebar.slider("↕️ Stride", min_value=1, max_value=4, value=1, step=1)
    stride_tuple = (stride, stride)
    
    # Padding
    padding_option = st.sidebar.selectbox("🛡️ Padding", options=["valid", "same"], index=1)
    if padding_option == "same":
        padding = kernel_size // 2
    else:
        padding = 0

    # Output padding
    output_padding = st.sidebar.slider("➕ Output Padding", min_value=0, max_value=4, value=0, step=1)

    # Activation
    activation = st.sidebar.selectbox("🔌 Activation Function", options=["ReLU", "Sigmoid", "Tanh", "None"], index=0)

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
            mode = st.selectbox("🎨 Select Image Mode for Deconvolution", options=["RGB", "Grayscale"], index=0)
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

            # Button to apply Conv2DTranspose
            if st.button("🚀 Apply Conv2DTranspose"):
                with st.spinner('🔄 Applying Conv2DTranspose...'):
                    # Define Conv2DTranspose layer
                    in_channels = image_tensor.shape[1]
                    conv_transpose_layer = nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=num_filters,
                        kernel_size=kernel_size_tuple,
                        stride=stride_tuple,
                        padding=padding,
                        output_padding=output_padding
                    )
                    
                    # Initialize weights (optional)
                    nn.init.xavier_uniform_(conv_transpose_layer.weight)
                    if conv_transpose_layer.bias is not None:
                        nn.init.zeros_(conv_transpose_layer.bias)
                    
                    # Apply Conv2DTranspose
                    conv_transpose_layer.eval()  # Set to evaluation mode
                    with torch.no_grad():
                        output = conv_transpose_layer(image_tensor)
                    
                    # Apply activation if selected
                    if activation == "ReLU":
                        act_fn = nn.ReLU()
                        output = act_fn(output)
                    elif activation == "Sigmoid":
                        act_fn = nn.Sigmoid()
                        output = act_fn(output)
                    elif activation == "Tanh":
                        act_fn = nn.Tanh()
                        output = act_fn(output)
                    
                    # Move to CPU and convert to numpy
                    output_np = output.squeeze(0).cpu().numpy()  # C_out x H_out x W_out
                    
                    # Calculate output dimensions
                    input_height = image_tensor.shape[2]
                    input_width = image_tensor.shape[3]
                    out_height = calculate_output_size_conv2dtranspose(input_height, kernel_size, stride, padding, output_padding)
                    out_width = calculate_output_size_conv2dtranspose(input_width, kernel_size, stride, padding, output_padding)
                    
                    # Display deconvolution parameters and output tensor shape
                    st.markdown("### 📊 Conv2DTranspose Details")
                    st.markdown(f"- **Filter Size (Kernel):** {kernel_size_tuple}")
                    st.markdown(f"- **Number of Filters (Output Channels):** {num_filters}")
                    st.markdown(f"- **Stride:** {stride_tuple}")
                    st.markdown(f"- **Padding:** {padding}")
                    st.markdown(f"- **Output Padding:** {output_padding}")
                    st.markdown(f"- **Output Feature Map Size:** {out_width} x {out_height} pixels")
                    st.markdown(f"- **Final Tensor Shape:** [Batch Size: {output.shape[0]}, Channels: {output.shape[1]}, Height: {output.shape[2]}, Width: {output.shape[3]}]")
                    st.markdown("---")
                    
                    # Display feature maps
                    st.header("📊 Feature Maps")
                    
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
                        axes[i].set_title(f'Filter {i+1}')
                    
                    # Hide any unused subplots
                    for j in range(num_filters_to_display, len(axes)):
                        axes[j].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.success("✅ Conv2DTranspose applied successfully!")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.write("Please check the console for more details.")

    else:
        st.info("💡 Please upload an image to get started.")

if __name__ == "__main__":
    main()
