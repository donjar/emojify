import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
time.sleep(1) # time for camera

delay = 1 # save image every 1 second
last_saved = time.time()
index = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    if (time.time() - last_saved) > delay:
        last_saved = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('image-{}.png'.format(index), gray)
        print("Saved", 'image-{}.png'.format(index))
        index += 1

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
