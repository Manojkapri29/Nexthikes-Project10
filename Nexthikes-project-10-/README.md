# Custom OCR Project
YOLOv3 + Tesseract pipeline (starter repo)

# Custom OCR using YOLOv3 and Tesseract

## Project Overview  
This project builds a custom Optical Character Recognition (OCR) system that automatically reads specific parts of lab report images and converts them into editable text files. It uses YOLOv3 to detect important regions in the image and Tesseract OCR to extract text from those regions.

## How It Works  
- Train a custom YOLOv3 model on your dataset to detect key objects in lab reports.  
- Use the detected bounding boxes to crop relevant areas from the images.  
- Process the cropped images (resize, grayscale, blur, threshold) to improve reading accuracy.  
- Pass processed images to Tesseract OCR to extract text and save it as CSV files.

## Features  
- Custom object detection with YOLOv3 tailored for lab reports.  
- Image preprocessing pipeline to improve text recognition.  
- Outputs editable CSV files with extracted information.

## Getting Started  
1. Install Tesseract OCR engine on your system.  
2. Install required Python libraries:  
   ```
   pip install pytesseract opencv-python
   ```
3. Train YOLOv3 model using your dataset and save weights in a `model` folder.  
4. Run the main script with an image as input:  
   ```
   python CustomOCR.py --image yourimage.jpg
   ```

## Dataset  
Link to the dataset used for training is provided in the project files (Google Drive).

## Future Work  
- Improve model accuracy by fine-tuning training and preprocessing.  
- Automate deployment on cloud platforms like AWS SageMaker or Google Cloud.  
- Enable end-to-end pipeline for easier use and scalability.
