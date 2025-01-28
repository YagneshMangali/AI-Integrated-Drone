import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
prototxt_path = "deploy.prototxt"
model_path = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define the classes that the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# We are only interested in "person" and "car"
TARGET_CLASSES = ["person", "car"]

# Initialize the video stream
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

# Get the frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()
(h, w) = frame.shape[:2]

# Define the line position (horizontal line in the middle of the frame)
line_y = h // 2
line_color = (0, 255, 0)  # Green color
line_thickness = 2

# Initialize counters and tracking
human_count = 0
vehicle_count = 0
crossed_objects = set()  # To track objects that have already crossed the line

# Function to save counts to a file
def save_counts(human_count, vehicle_count):
    with open("crossed_counts.txt", "w") as file:
        file.write(f"Humans crossed: {human_count}\n")
        file.write(f"Vehicles crossed: {vehicle_count}\n")

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the horizontal line in the middle of the frame
    cv2.line(frame, (0, line_y), (w, line_y), line_color, line_thickness)

    # Prepare the frame for object detection by creating a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            # Check if the detected object is a person or a car
            if CLASSES[idx] in TARGET_CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate the center of the bounding box
                center_x = (startX + endX) // 2
                center_y = (startY + endY) // 2

                # Check if the object has crossed the line
                if center_y > line_y and i not in crossed_objects:
                    crossed_objects.add(i)  # Mark this object as crossed
                    if CLASSES[idx] == "person":
                        human_count += 1
                    elif CLASSES[idx] == "car":
                        vehicle_count += 1

    # Display the counts on the top right corner of the frame
    cv2.putText(frame, f"Humans crossed: {human_count}", (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Vehicles crossed: {vehicle_count}", (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Real-time Object Detection", frame)

    # Save the counts to a file
    save_counts(human_count, vehicle_count)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()