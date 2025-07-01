# Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management
This project aims to develop a Transfer learning-based system for classifying poultry diseases into four categories: Salmonella, New Castle Disease, Coccidiosis, and Healthy. The solution involves creating a robust machine learning model that will be integrated into a mobile application. Farmers will be able to use this application 
Transfer Learning Based Classification of Poultry Diseases

This project applies transfer learning using the EfficientNetB0 model to classify poultry diseases into four categories

- Salmonella  
- Newcastle Disease  
- Coccidiosis  
- Healthy

The final system is intended for use in a mobile application that allows farmers to upload images and receive real-time predictions. This supports early disease detection and improves poultry health management.

Project Objectives

- Build an accurate deep learning model using transfer learning  
- Classify poultry images into specific disease classes  
- Integrate the model into a user-friendly mobile interface  
- Improve early diagnosis and reduce bird mortality

Dataset

- Source Kaggle Chicken Disease Dataset  
- The dataset contains poultry images categorized into four classes  
- Data is preprocessed including resizing to 224 by 224 pixels and augmentation such as flip zoom and shear

Model Architecture

- Base Model EfficientNetB0 pretrained on ImageNet  
- Custom classification layers include  
  - Global average pooling  
  - Dropout layer with 0.5 rate  
  - Dense output layer using softmax activation  

- Loss Function categorical crossentropy  
- Optimizer Adam  
- Base layers are frozen to retain prelearned features

Training Setup

- Image size 224 x 224  
- Batch size 32  
- Epochs 10  
- Validation split 20 percent  
- Augmentation includes horizontal flip zoom and shear

Results

- Model training and validation accuracy are tracked using matplotlib  
- Accuracy graphs help visualize performance and detect overfitting or underfitting

Mobile App Integration

- The trained model can be deployed into a mobile application  
- Farmers can upload images for instant disease classification  
- The system provides disease-specific feedback to assist in treatment and care

Installation and Setup

To install required libraries use

pip install tensorflow matplotlib kaggle

Upload your kaggle dot json credentials to access the dataset and run the training script

License

This project is licensed under the MIT License

Contributing

Contributions are welcome Fork the repository and create a pull request

Acknowledgements

- Kaggle contributor Allan Dclive for providing the chicken disease dataset  
- TensorFlow and Keras for enabling deep learning development

