DEEP LEARNING BASED REAL-TIME IMAGE CLASSIFICATION
README

1. Project Description
This project is a deep learning-based real-time image classification system. 
A pre-trained AlexNet convolutional neural network (CNN) is used and fine-tuned with transfer learning on a custom dataset. 
The trained model can classify objects both offline and in real time using a webcam.

2. Requirements

Software Requirements:
- MATLAB (R2020a or later recommended)
- MATLAB Deep Learning Toolbox
- MATLAB Image Processing Toolbox
- Webcam support package (for real-time testing)

Hardware Requirements:
- Webcam (for live classification)
- GPU (optional but recommended for faster training)

3. Dataset Structure

The dataset must be organized in a folder-based structure.
Each folder represents one class label.

Example structure:

dataset/
│
├── bottle/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
├── headphone/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
└── phone/
    ├── img1.jpg
    ├── img2.jpg
    └── ...

To add a new class, create a new folder and place relevant images inside it.
The system automatically detects folder names as class labels.

4. Project Files

- egitim.m
  This script loads the dataset, splits the data into training and test sets, 
  modifies the AlexNet model using transfer learning, and trains the network.

- canli_tahmin.m
  This script loads the trained model and performs real-time object classification using a webcam.

5. Training the Model

Steps to train the model:

1. Place your dataset in the correct folder structure.
2. Open MATLAB and set the project folder as the working directory.
3. Run the script:

   egitim.m

During training:
- Images are resized to 227x227x3.
- Data is split into 80% training and 20% testing.
- AlexNet is fine-tuned by replacing the final layers.
- The model is trained using SGDM optimizer.

After training, the trained network is saved for later use.

6. Real-Time Classification

Steps for live testing:

1. Make sure the trained model file is available.
2. Connect a webcam to your computer.
3. Run the script:

   canli_tahmin.m

The system captures frames from the webcam and classifies objects in real time.
Prediction labels and confidence scores are displayed on the screen.

7. Confidence Threshold

A confidence threshold of 60% is used.
- If confidence >= 60%, the prediction is accepted and displayed.
- If confidence < 60%, the result is labeled as "UNCERTAIN".

8. Evaluation

Model performance is evaluated using:
- Test accuracy
- Confusion matrix

The confusion matrix shows classification performance.
Correct predictions are concentrated on the diagonal, indicating high accuracy.

9. Notes

- Increasing the number of images per class improves model accuracy.
- Balanced datasets (similar number of images per class) are recommended.
- GPU acceleration significantly reduces training time.

10. A
