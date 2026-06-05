Project 4: Image or Text Recognition Pipeline (Basic)

Description:
This project implements a visual recognition pipeline using OpenCV and Tesseract OCR.
It performs:
1. Object Detection: Detects objects (like person and dog) in the image using MobileNet-SSD via OpenCV's DNN module.
2. Text Recognition (OCR): Uses Google's Tesseract OCR to extract text from images.

Files:
- Artificial_intelligence_Project4.py: The main Python script.
- hiking_dog_mountain.avif: The input image.
- MobileNetSSD_deploy.prototxt: Model architecture file (automatically downloaded).
- mobilenet_iter_73000.caffemodel: Model weights file (automatically downloaded/placed).
- synthetic_ocr_verification.png: High-contrast test image created to verify the OCR engine.
- recognized_output.png: The visual output image showing bounding boxes and OCR results.

Prerequisites:
Make sure Python is installed.
Install the required Python packages:
pip install opencv-python numpy pytesseract pillow pillow-heif

Tesseract OCR Installation:
Google's Tesseract OCR engine must be installed on your system.
On Windows, you can install it using PowerShell:
winget install UB-Mannheim.TesseractOCR --silent --accept-source-agreements --accept-package-agreements

By default, the installer places Tesseract in "C:\Program Files\Tesseract-OCR\tesseract.exe". The script is configured to look for Tesseract at this location.

How to Run:
Run the script using:
python Artificial_intelligence_Project4.py

Expected Outputs:
The script will download the model files if they are not present, perform object detection and OCR, print the results in the terminal, and save the visualization as "recognized_output.png".
