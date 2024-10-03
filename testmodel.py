import cv2

# Initialize the video capture
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the LBPH face recognizer and the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# List of names corresponding to face IDs (index 0 is unused)
name_list = ["", "Shrey"]

# Start capturing and recognizing faces
while True:
    # Capture frame from the video feed
    ret, frame = video.read()

    # Convert the frame to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(grey, 1.3, 5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Predict the face ID and confidence level
        serial, conf = recognizer.predict(grey[y:y+h, x:x+w])

        if conf > 50:
            # If confidence is high, display the recognized name
            cv2.putText(frame, name_list[serial], (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        else:
            # If confidence is low, display "Unknown Person"
            cv2.putText(frame, "Unknown Person", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    # Display the frame with the recognition results
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

print("Dataset Collection Done!!!!!!!!!")
