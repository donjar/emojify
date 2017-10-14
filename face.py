import numpy as np
import cv2
from sys import argv

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

if len(argv) == 1:
    print('Usage: python {} IMAGE_LOCATION'.format(argv[0]))
    exit(1)

img = cv2.imread(argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect face. The input must be a grayscale image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # Draw a blue rectangle on the face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Detect eyes. The input must be a grayscale image
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        # Draw a green rectangle on the face
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
