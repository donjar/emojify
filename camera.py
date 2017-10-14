import numpy as np
import cv2
import time
import matplotlib.image as image

import emotion as classifier

cap = cv2.VideoCapture(0)
time.sleep(1) # time for camera

delay = 1 # save image every 1 second
last_saved = time.time()
index = 1

# emoji stuff
emoji = cv2.imread('emojis/dizzy_face.png')
# print(np.array_equal(emoji[0][0], [[0, 0, 0]]))

# load emotion classifier
c = classifier.load_classifier('model')

emoji_mappings = {'anger':'steaming', 'disgust':'dizzy_face', 'fear':'scream',
                  'happy':'triumph', 'sadness':'slight_frown', 'surprise':'flushed'}


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    if (time.time() - last_saved) > delay:
        last_saved = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('temp/image-{}.png'.format(index), gray)

        # attempt to read emotion
        result = classifier.predict_image(c, 'temp/image-{}.png'.format(index))
        if result != -1:
            print("emotion", result)
            emoji = cv2.imread('emojis/' + emoji_mappings[result] + '.png')
        index += 1


    s_img = emoji
    l_img = frame
    y_offset = x_offset = 300
    for y in range(s_img.shape[0]):
        for x in range(s_img.shape[1]):
            if not np.array_equal(s_img[y][x], emoji[0][0]):
                l_img[y_offset + y, x_offset + x] = s_img[y][x]

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
