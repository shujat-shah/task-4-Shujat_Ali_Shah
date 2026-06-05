import os
import sys
import urllib.request
import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
CAFFEMODEL_URL = "https://cdn.jsdelivr.net/gh/chuanqi305/MobileNet-SSD@master/mobilenet_iter_73000.caffemodel"

PROTOTXT_FILE = "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_FILE = "mobilenet_iter_73000.caffemodel"

# PASCAL VOC classes detected by MobileNet-SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Color palette for classes (BGR format for OpenCV)
CLASS_COLORS = {
    "person": (240, 100, 36),   # Sleek Orange/Red
    "dog": (36, 220, 120),      # Vibrant Emerald Green
    "default": (180, 105, 255)  # Soft Pink/Purple
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def print_banner(title):
    print("=" * 70)
    print(f" {title.center(68)} ")
    print("=" * 70)

def download_file(url, filename):
    """Downloads a file from a URL with basic progress reporting and custom headers."""
    if os.path.exists(filename):
        print(f"[Info] '{filename}' already exists, skipping download.")
        return True
    
    print(f"[Download] Fetching {filename} from {url}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            totalsize = int(response.info().get('Content-Length', 0))
            readsofar = 0
            blocksize = 1024 * 1024 # 1MB blocks
            
            while True:
                data = response.read(blocksize)
                if not data:
                    break
                out_file.write(data)
                readsofar += len(data)
                if totalsize > 0:
                    percent = readsofar * 100 / totalsize
                    sys.stdout.write(f"\r[Download] Progress: {percent:5.1f}% ({readsofar/1024/1024:.2f} MB / {totalsize/1024/1024:.2f} MB)")
                else:
                    sys.stdout.write(f"\r[Download] Read: {readsofar/1024/1024:.2f} MB")
                sys.stdout.flush()
            print(f"\n[Download] Successfully saved to {filename}")
        return True
    except Exception as e:
        print(f"\n[Error] Failed to download {filename}: {e}")
        # Clean up partial download if it exists
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
        return False

def configure_tesseract():
    """Locates and configures Tesseract OCR path on Windows."""
    # Check if tesseract is already in environment PATH
    try:
        pytesseract.get_tesseract_version()
        print("[Tesseract] OCR engine detected in system PATH.")
        return True
    except pytesseract.TesseractNotFoundError:
        pass

    # Common Windows installation directories
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    # Check local app data just in case
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        common_paths.append(os.path.join(local_app_data, "Programs", "Tesseract-OCR", "tesseract.exe"))

    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"[Tesseract] OCR path manually configured to: {path}")
            return True

    print("[Warning] Tesseract OCR executable not found. OCR features might fail.")
    return False

# ==========================================
# RECOGNITION PIPELINE CLASS
# ==========================================
class VisionRecognitionPipeline:
    def __init__(self):
        configure_tesseract()
        self.net = None
        self.load_object_detection_model()

    def load_object_detection_model(self):
        """Downloads and loads the MobileNetSSD model."""
        download_file(PROTOTXT_URL, PROTOTXT_FILE)
        download_file(CAFFEMODEL_URL, CAFFEMODEL_FILE)
        
        if os.path.exists(PROTOTXT_FILE) and os.path.exists(CAFFEMODEL_FILE):
            print("[DNN] Loading MobileNet-SSD model...")
            self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILE, CAFFEMODEL_FILE)
            print("[DNN] Model loaded successfully.")
        else:
            print("[Error] MobileNet-SSD model files are missing or could not be downloaded.")

    def load_image(self, path):
        """Loads an image (including AVIF format) and converts it to BGR OpenCV format."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at path: {path}")
            
        print(f"[Image] Loading image: {path}")
        # Use Pillow to handle AVIF files since OpenCV might not support them natively
        try:
            pil_img = Image.open(path)
            # Convert RGB to BGR
            cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            print(f"[Image] Image successfully loaded. Size: {cv_image.shape[1]}x{cv_image.shape[0]}")
            return cv_image
        except Exception as e:
            raise RuntimeError(f"Failed to load image using Pillow: {e}")

    def run_object_detection(self, cv_image, conf_threshold=0.2):
        """Runs MobileNet-SSD on the image and returns detections."""
        if self.net is None:
            print("[Warning] Object detection net is not loaded. Skipping detection.")
            return []

        h, w = cv_image.shape[:2]
        # Preprocessing: resize to 300x300, scale factor 0.007843 (1/127.5), mean subtraction 127.5
        blob = cv2.dnn.blobFromImage(cv_image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        # detections shape is [1, 1, N, 7]
        num_detections = detections.shape[2]
        
        for i in range(num_detections):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                class_id = int(detections[0, 0, i, 1])
                class_name = CLASSES[class_id]
                
                # Compute bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image boundaries and cast to standard Python ints
                startX = int(max(0, startX))
                startY = int(max(0, startY))
                endX = int(min(w - 1, endX))
                endY = int(min(h - 1, endY))
                
                results.append({
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "box": (startX, startY, endX, endY)
                })
                
        return results

    def run_ocr(self, cv_image):
        """Performs text recognition using pytesseract."""
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to improve OCR accuracy
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except Exception as e:
            return f"OCR Error: {e}"

    def create_synthetic_text_image(self):
        """Generates a synthetic image containing text to verify the OCR engine."""
        print("[OCR Setup] Generating synthetic high-contrast text image to verify Tesseract...")
        
        width, height = 600, 200
        # White background
        pil_img = Image.new("RGB", (width, height), color=(255, 255, 255))
        d = ImageDraw.Draw(pil_img)
        
        # Try to use a nice font, fallback to default PIL font
        font = None
        for font_name in ["arial.ttf", "calibri.ttf", "cour.ttf", "msyh.ttc"]:
            try:
                # System fonts folder on Windows
                font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", font_name)
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 28)
                    break
            except Exception:
                continue
                
        if font is None:
            font = ImageFont.load_default()
            
        # Draw high-contrast text
        text_line1 = "TESSERACT OCR SYSTEM"
        text_line2 = "Machine perception is fully active!"
        text_line3 = "Confidence Level: 99.4%"
        
        d.text((20, 25), text_line1, fill=(10, 10, 10), font=font)
        d.text((20, 75), text_line2, fill=(36, 120, 240), font=font) # Sleek blue accent
        d.text((20, 125), text_line3, fill=(60, 160, 60), font=font) # Green accent
        
        # Convert back to cv2 BGR format
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return cv_img

    def draw_detections(self, image, detections):
        """Draws bounding boxes and labels on the image."""
        annotated_image = image.copy()
        
        for det in detections:
            name = det["class_name"]
            conf = det["confidence"]
            (startX, startY, endX, endY) = det["box"]
            
            # Determine color
            color = CLASS_COLORS.get(name, CLASS_COLORS["default"])
            
            # Draw primary bounding box
            cv2.rectangle(annotated_image, (startX, startY), (endX, endY), color, 3)
            
            # Draw nice text banner for label
            label = f"{name.upper()}: {conf:.1%}"
            
            # Calculate text size for background banner
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            
            # Label position (adjust if near the top boundary)
            text_y = startY - 10 if startY - 10 > 25 else startY + text_h + 10
            
            # Draw label banner background
            cv2.rectangle(
                annotated_image, 
                (startX - 2, text_y - text_h - 6), 
                (startX + text_w + 10, text_y + baseline), 
                color, 
                cv2.FILLED
            )
            
            # Draw label text in white
            cv2.putText(
                annotated_image, 
                label, 
                (startX + 4, text_y - 2), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.7, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
        return annotated_image

    def create_dashboard(self, annotated_image, original_ocr, synthetic_ocr):
        """Creates a beautiful combined visual dashboard showcasing all outputs."""
        h, w = annotated_image.shape[:2]
        
        # We will create an informational bottom panel for the OCR results
        panel_h = 240
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
        # Background color: modern dark grey (glassmorphism/sleek look)
        panel[:] = (32, 28, 24) 
        
        # Draw borders and title
        cv2.rectangle(panel, (0, 0), (w-1, panel_h-1), (80, 80, 80), 2)
        cv2.line(panel, (0, 45), (w-1, 45), (80, 80, 80), 1)
        
        # Panel Title
        cv2.putText(panel, "MACHINE PERCEPTION DASHBOARD - INTEL & RECOGNITION", (20, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Content formatting
        original_ocr_clean = original_ocr.replace("\n", " | ") if original_ocr else "NO TEXT DETECTED IN ORIGINAL IMAGE"
        synthetic_ocr_clean = synthetic_ocr.replace("\n", " | ")
        
        # Draw Original OCR Status
        cv2.putText(panel, "[SYSTEM] Pytesseract OCR - Original Visual Source:", (20, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Result: {original_ocr_clean}", (40, 110), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (140, 190, 255), 1, cv2.LINE_AA)
                    
        # Draw Synthetic OCR Verification Status
        cv2.putText(panel, "[SYSTEM] Pytesseract OCR - Synthetic Verification Source:", (20, 160), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Result: {synthetic_ocr_clean}", (40, 190), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (120, 255, 170), 1, cv2.LINE_AA)

        # Vertical separator line on the right side if there's enough space
        # Connect everything vertically
        dashboard = np.vstack([annotated_image, panel])
        return dashboard

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    print_banner("PROJECT 4: IMAGE & TEXT RECOGNITION PIPELINE")
    
    # Initialize pipeline
    pipeline = VisionRecognitionPipeline()
    
    # 1. Load User's Visual Source
    input_image_path = "hiking_dog_mountain.avif"
    try:
        image = pipeline.load_image(input_image_path)
    except Exception as e:
        print(f"[Critical Error] Failed to load target image: {e}")
        return
        
    # 2. Run Object Recognition
    print("\n[Analysis] Running MobileNet-SSD Object Detector...")
    detections = pipeline.run_object_detection(image, conf_threshold=0.2)
    
    print("-" * 50)
    print(f"Detected Objects ({len(detections)} total):")
    for idx, det in enumerate(detections, 1):
        box = det["box"]
        print(f"  {idx}. Class: '{det['class_name']}' | Confidence: {det['confidence']:.2%} | Box: {box}")
    print("-" * 50)
    
    # 3. Run OCR on Original Image
    print("\n[Analysis] Running Pytesseract OCR on Original Image...")
    original_ocr_text = pipeline.run_ocr(image)
    print(f"Extracted OCR Text:\n>>> {repr(original_ocr_text)}")
    
    # 4. Generate Synthetic Text Image and Verify OCR
    print("\n[Analysis] Running OCR Verification on Synthetic Image...")
    synthetic_image = pipeline.create_synthetic_text_image()
    synthetic_ocr_text = pipeline.run_ocr(synthetic_image)
    print(f"Extracted OCR Verification Text:\n>>> {repr(synthetic_ocr_text)}")
    
    # 5. Save the Verification Image
    cv2.imwrite("synthetic_ocr_verification.png", synthetic_image)
    print("[Saved] Saved synthetic verification source as 'synthetic_ocr_verification.png'.")
    
    # 6. Annotate Image and Create Final Combined Dashboard
    print("\n[Visualization] Annotating bounding boxes and combining dashboard...")
    annotated_img = pipeline.draw_detections(image, detections)
    dashboard_img = pipeline.create_dashboard(annotated_img, original_ocr_text, synthetic_ocr_text)
    
    # 7. Output Final Visual Deliverable
    output_filename = "recognized_output.png"
    cv2.imwrite(output_filename, dashboard_img)
    print(f"[Success] Pipeline output saved to current directory as: '{output_filename}'")
    
    print("\n" + "=" * 70)
    print(" PIPELINE RUN COMPLETE - DEMO SUCCESSFUL ".center(70, "="))
    print("=" * 70)

if __name__ == "__main__":
    main()