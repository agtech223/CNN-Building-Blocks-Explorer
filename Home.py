import streamlit as st

def main():
    st.set_page_config(page_title="CNN Building Blocks Explorer", layout="wide")
    
    # Header
    st.title("🧠 CNN Building Blocks Explorer")
    
    # Subtitle and Introduction
    st.markdown("""
    ### Explore the Core Operations of Convolutional Neural Networks!
    
    Welcome to the **CNN Building Blocks Explorer**! This application provides an interactive way to understand the fundamental building blocks of Convolutional Neural Networks (CNNs) using PyTorch. 
    Upload your images and apply operations to gain an intuitive understanding of CNNs. Whether you're just starting with deep learning or want a tool to visualize CNN layers, this app is here for you.
    """)
    
    # Section for Application Features
    st.markdown("---")
    st.header("🔍 What You Can Do with This App")
    st.markdown("""
    - **Convolution (Conv2D)**: Apply convolutional filters to your images and observe feature extraction in action.
    - **Normalization (BatchNorm2D)**: See how normalization stabilizes and speeds up model training by scaling your data.
    - **Transpose Convolution (Conv2DTranspose)**: Visualize upsampling, often used in applications like image segmentation and generation.
    - **MaxPooling (MaxPool2D)**: Downsample your images, highlighting dominant features while reducing spatial dimensions.
    """)
    
    # Navigation Info
    st.markdown("---")
    st.header("🔗 Navigation")
    st.write("Use the sidebar to select and explore each operation.")
    
    # Example Workflow Section
    st.markdown("---")
    st.header("🚀 Getting Started")
    st.markdown("""
    1. **Upload an Image**: Choose any image from your device.
    2. **Choose an Operation**: Select from Convolution, Normalization, Transpose Convolution, or MaxPooling.
    3. **Adjust Parameters**: Customize parameters for each operation (like filter size, stride, padding).
    4. **Visualize the Results**: Instantly see how the operation modifies your image and learn more about each layer's role in CNNs.
    """)

    # About the Technology
    st.markdown("---")
    st.header("📘 About Convolutional Neural Networks")
    st.markdown("""
    CNNs are at the heart of many computer vision applications, from object detection to image classification. They consist of various types of layers, each playing a unique role in processing and understanding visual data. Here's a quick overview of the operations this app lets you explore:
    
    - **Convolution**: Extracts features by sliding filters over the image.
    - **Batch Normalization**: Normalizes data across the batch to stabilize training.
    - **Transpose Convolution**: Upsamples data, often used in tasks like image segmentation.
    - **Max Pooling**: Downsamples data by selecting maximum values within each region.
    
    Experimenting with these layers can deepen your understanding of how CNNs work!
    """)

    # Footer with Developer Information and Credits
    st.markdown("---")
    st.write("Made with ❤️ using [Streamlit](https://streamlit.io/) and [PyTorch](https://pytorch.org/).")
    
    st.markdown("""
    ---
    **Developer Information**  
    - **Name**: Hassan Afzaal, EIT, M.Sc  
    - **Email**: [h.afzaal4242@gmail.com](mailto:h.afzaal4242@gmail.com)  
    - **LinkedIn**: [linkedin.com/in/hassan-afzaal-eit-230451195/](https://www.linkedin.com/in/hassan-afzaal-eit-230451195/)  
    """)

if __name__ == "__main__":
    main()
