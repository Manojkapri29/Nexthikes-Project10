import cv2
import pytesseract
import numpy as np

def run_ocr(image_path):
    # Load image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Load YOLOv3 model
    net = cv2.dnn.readNet('models/yolov3.weights', 'cfg/yolov3.cfg')
    layer_names = net.getUnconnectedOutLayersNames()
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    # Parse YOLO outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    results = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        roi = image[y:y+h, x:x+w]

        # Preprocessing for OCR
        roi = cv2.resize(roi, None, fx=3, fy=3)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(thresh)

        # OCR
        text = pytesseract.image_to_string(inverted)
        results.append(text)

    return results