import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Get output layer indices
output_layers_indices = net.getUnconnectedOutLayers().flatten().tolist()

# Get the names of all layers in the network
layer_names = net.getLayerNames()

# Extract the names of the output layers from the layer names
output_layers = [layer_names[idx - 1] for idx in output_layers_indices]

# Read the class names from coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Read the video file
cap = cv2.VideoCapture("video2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            # Check if class_id is within the range of classes
            if 0 <= class_id < len(classes):
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":
                    # Draw a green rectangle around the person
                    x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Show the processed frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):  # Reduce the delay to 10 milliseconds
        break

cap.release()
cv2.destroyAllWindows()
