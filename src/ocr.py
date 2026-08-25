import cv2
from paddleocr import PaddleOCR

# Issues arise when using PaddleOCR in the same environment as YOLO
# Use different environments for the two

def main():
    ocr = PaddleOCR(
        lang="en",
        device="gpu:0" # If using multi-gpu system
    )

    # Test with one cropped photo
    img_path = "dataset/crops/IMG_5010_sign_0.jpg"
    result = ocr.predict(img_path)

    # Gives only the text, could also add the confidence and the rectangles too for testing
    for item in result[0]['rec_texts']:
        print(item)

def preprocess_ocr(image):
    # PREPROCESS IMG
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    # Denoise (reduce noise for cleaner edges)  
    denoised = cv2.medianBlur(gray, ksize=3)  
    
    # Binarize: Black text on white background (adjust threshold if text is light)  
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  
    # THRESH_BINARY_INV inverts colors: ensures text is white (255) on black (0) background

    return

def ocr_text(image) -> tuple[str, float]:  
    # Configure OCR to read in text (maybe paddle ocr or whichever works best)

    # TODO
    text = text.strip()
    text = text.upper()
    return

if __name__ == "__main__":
    main()