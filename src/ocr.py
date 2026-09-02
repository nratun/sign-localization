import cv2
from paddleocr import PaddleOCR

# Issues arise when using PaddleOCR in the same environment as YOLO
# Use different environments for the two or use cpu rather than gpu

def preprocess_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    # Denoise
    denoised = cv2.medianBlur(gray, ksize=3)  
    
    # Binarize: Black text on white background
    # TODO sign is split into two regions (black text/light bg, light text/black bg)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

    return thresh

def ocr_text(ocr, image) -> tuple[str, float]:
    # Preprocess image (not using for now)
    # processed = preprocess_ocr(image)

    # Run OCR
    result = ocr.predict(image)

    if not result:
        return "", 0.0

    result = result[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])

    if not texts:
        return "", 0.0
    
    # Combine all detected text
    text = " ".join(texts)
    text = text.strip().upper()

    # Calculate average confidence
    if scores:
        conf = sum(scores) / len(scores)
    else:
        conf = 0.0

    return text, conf

def main():
    ocr = PaddleOCR(
        lang="en",
        device="cpu", # If using multi-gpu system gpu:0
        enable_mkldnn=False
    )

    # Test image
    img_path = "dataset/crops/IMG_5024_sign_0.jpg"

    image = cv2.imread(img_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {img_path}"
        )

    # Run OCR
    text, confidence = ocr_text(
        ocr,
        image
    )

    print(f"Text: {text}")
    print(f"Confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()