import cv2
import numpy as np
import glob

# --- CONFIGURATION ---
WEIGHTS_PATH = "backup/yolov3-custom_final.weights"
CONFIG_PATH = "cfg/yolov3-custom.cfg"
NAMES_PATH = "data/obj.names"
IMAGE_PATH = "data/test_image.jpg"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# --- LOAD THE MODEL ---
print("Loading YOLO model...")
net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
classes = []
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("Model loaded successfully.")

# --- LOAD AND PROCESS THE IMAGE ---
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not read image at {IMAGE_PATH}")
    exit()

height, width, channels = img.shape

# Create a blob from the image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# --- PROCESS DETECTIONS ---
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > CONF_THRESHOLD:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Max Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

# --- DRAW BOUNDING BOXES ---
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence}", (x, y + 20), font, 2, color, 2)

# --- SHOW AND SAVE RESULT ---
cv2.imshow("Image", img)
cv2.imwrite("prediction_result.jpg", img)
print("Prediction result saved as prediction_result.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()
