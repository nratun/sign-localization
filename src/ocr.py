import cv2

def ocr_text(image) -> str:  
    # Configure OCR to read in text (maybe paddle ocr or whichever works best)

    # PREPROCESS IMG
    # Convert to grayscale  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    # Denoise (reduce noise for cleaner edges)  
    denoised = cv2.medianBlur(gray, ksize=3)  
    
    # Binarize: Black text on white background (adjust threshold if text is light)  
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  
    # THRESH_BINARY_INV inverts colors: ensures text is white (255) on black (0) background

    # TODO
    return