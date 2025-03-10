# Road Surface Semantic Segmentation for Autonomous Navigation

## Overview

This project aims to improve the visual perception for autonomous navigation by performing pixel-wise semantic segmentation of road surfaces, especially in low-resolution images. A Convolutional Neural Network (CNN) is employed to identify and categorize road surfaces, focusing on detecting surface variations, damages, and unpaved roads. The U-Net architecture is used for semantic segmentation, with ResNet34 and ResNet50 models from the FastAI library as backbone architectures. The project utilizes transfer learning (TL) for fine-tuning the models on the **RTK dataset** from Brazil, which provides a diverse range of road conditions.

## Key Objectives

- **Road Surface Classification**: Identify and classify road types, including damaged and unpaved roads.
- **Semantic Segmentation**: Perform pixel-wise segmentation of road surfaces, categorizing different regions of the road.
- **Low-Resolution Image Handling**: Process and segment low-resolution images accurately.
- **Model Optimization**: Achieve optimal performance with strategies like Transfer Learning, Fine-Tuning, and Data Augmentation.

## Methodology

1. **Dataset**:
   - We use the **RTK dataset** from Brazil, which contains diverse surface types, including damages, and unpaved roads. This dataset provides a more realistic set of challenges for the model compared to other datasets with well-maintained roads.

2. **Model Architecture**:
   - **U-Net**: The U-Net architecture is chosen due to its success in semantic segmentation tasks. The encoder of U-Net is based on **ResNet34** and **ResNet50**, which act as feature extractors. The decoder performs upsampling, and **skip-connections** are used to preserve vital low-level features.
   - **ResNet Backbone**: Pretrained **ResNet34** and **ResNet50** models are used for feature extraction. Transfer learning is applied, with the models pretrained on **ImageNet** and fine-tuned on the RTK dataset.
   - **Training Strategy**: The **One Cycle Policy** is used to dynamically adjust the learning rate during training, allowing for better convergence.

3. **Data Augmentation**:
   - Data augmentation techniques are applied to both original images and their respective segmentation masks to improve generalization. This includes rotations, flips, and other geometric transformations.

4. **Training**:
   - The model is trained using the **FastAI** library, which automates the creation of the decoder and enables the use of pretrained ResNet backbones.
   - Transfer Learning and Fine-Tuning (TL/FT) are performed using pretrained weights from ImageNet to accelerate convergence.

5. **Model Performance**:
   - After training, the model achieved an accuracy of **97.75%** using the ResNet34 backbone.

## Streamlit Interface

### Introduction

The project also features an interactive **Streamlit** user interface to visualize the semantic segmentation results. Streamlit is a powerful and easy-to-use framework for creating data-driven web applications. With this interface, users can upload road images and immediately see the segmented output of the road surfaces, helping to assess the effectiveness of the model in real-time.

### Streamlit Features:
- **Image Upload**: Users can upload an image directly through the interface.
- **Dynamic Segmentation**: Once an image is uploaded, the model processes the image and generates a segmented image displaying different road surface categories.
- **Interactive Visuals**: The application allows real-time visualization of results, offering a clear view of how the model interprets the road surface in the image.
- **Model Insights**: Users can explore various images and their corresponding segmented outputs, enabling a better understanding of the model's accuracy and performance.

### How the Streamlit Interface Works:
- Upon uploading an image through the Streamlit interface, the system runs the segmentation model and outputs a segmented image. This segmentation shows pixel-wise classifications, distinguishing road types such as asphalt, damaged roads, and unpaved surfaces.
- The interface is designed to be intuitive and user-friendly, making it accessible to non-experts who want to test the model on different road images.

## Dependencies

- Python 3.x
- **FastAI** (for model training)
- **Streamlit** (for the web interface)
- **PyTorch** (for deep learning)
- **OpenCV**
- **NumPy**
- **Matplotlib**
## Acknowledgements

- **U-Net**: For semantic segmentation tasks, especially in biomedical and image analysis domains.
- **ResNet**: For utilizing powerful pre-trained models that help speed up the training process and improve model accuracy.
- **RTK Dataset**: For providing a diverse set of real-world road images with various surface types and conditions, allowing for better generalization of the model.
