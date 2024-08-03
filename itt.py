import os

import cv2
import numpy as np
import pytesseract

from PIL import Image


os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/tessdata/"
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Apply dilation to remove noise
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)

    # Apply erosion to remove noise
    gray = cv2.erode(gray, kernel, iterations=1)

    # Apply median blur to remove noise
    gray = cv2.medianBlur(gray, 3)

    return Image.fromarray(gray)


def ocr_core(image):
    text = pytesseract.image_to_string(
        image, lang="eng+hin"
    )
    return text


if __name__ == "__main__":
    image_path = "samples/input/text/page_1.png"
    preprocessed_image = preprocess_image(image_path)
    text = ocr_core(preprocessed_image)
    with open("samples/output/text/page_1.txt", "w", encoding="utf-8") as file:
        file.write(text)
