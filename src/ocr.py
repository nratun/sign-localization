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

def ocr_text(ocr, image) -> list[dict]:
    # Preprocess image (not using for now)
    # processed = preprocess_ocr(image)

    # Run OCR
    result = ocr.predict(image)

    if not result:
        return []

    result = result[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    coords = result.get("rec_polys", [])

    detections = []

    # Example format:
    # { "text": "397", "conf": 0.97, "box": np.array([[12, 8], [30, 8], [30, 20], [12, 20]]) }
    for text, score, points in zip(texts, scores, coords):
        detections.append({
            "text": text,
            "conf": float(score),
            "points": points
        })

    return detections