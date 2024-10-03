import cv2

# Initialize the video capture
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Number of users you want to collect data for
num_users = 3

# Loop to collect data for multiple users
for user in range(1, num_users + 1):
    id = input(f"Enter Face ID for User {user}: ")
    count = 0

    while True:
        # Capture frame from the video feed
        ret, frame = video.read()

        # Convert the frame to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = facedetect.detectMultiScale(grey, 1.3, 5)

        # Loop over the detected faces
        for (x, y, w, h) in faces:
            count += 1
            # Save the face image in a dataset folder
            cv2.imwrite(f'datasets/User.{id}.{count}.jpg', grey[y:y+h, x:x+w])
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        # Display the frame with the rectangle drawn around the face
        cv2.imshow(f"Collecting Data for User {id}", frame)

        # Stop collecting after pressing 'q' or collecting 3000 images
        k = cv2.waitKey(1)
        if count >= 3000:
            print(f"Dataset Collection Done for User {id}!")
            break

# Release the video and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
