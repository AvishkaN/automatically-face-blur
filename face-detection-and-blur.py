import cv2

# Load the DNN face detection model
model_path = "opencv_face_detector_uint8.pb"
config_path = "opencv_face_detector.pbtxt"

# Initialize the DNN model
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
import time

source='face-video-2.mp4'
source=0
cap = cv2.VideoCapture(source)

# Set the desired resolution
desired_width = 1280  # Example: 1280 pixels wide
desired_height = 720  # Example: 720 pixels high


face_counter=0
prev_time=0

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (desired_width, desired_height))
    if not ret:
        break

    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Prepare the frame as input to the DNN
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)

    # Run the model to detect faces
    detections = net.forward()

    # Process detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter detections by confidence level
        if confidence > 0.5:  # Adjust threshold as needed
            box = detections[0, 0, i, 3:7] * [width, height, width, height]
            (x, y, x1, y1) = box.astype("int")


            face = frame[y:y1, x:x1]



            if(len(face)):
                blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                face=blurred_face
                frame[y:y1, x:x1] = blurred_face


            face = frame[y:y1, x:x1]

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)  # FPS = 1 / time between frames
            prev_time = curr_time

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,0,255), 2)

    # Display the frame with detections
    cv2.imshow("Face Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
