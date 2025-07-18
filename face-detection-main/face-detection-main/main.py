import cv2

# Load OpenCV’s pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# For static images
IMAGE_FILES = ['test1.jpg', 'test2.jpg']  # Put your image file names here

for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    if image is None:
        print(f"Could not read {file}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(f'annotated_image_{idx}.png', image)
    print(f"Saved annotated_image_{idx}.png")

# For webcam input
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (y + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection (OpenCV)', frame)

    # Exit when ESC is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
