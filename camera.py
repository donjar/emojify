import numpy as np
import cv2
import time
import pyperclip
from face_detect import find_faces
from PIL import Image

import emotion as classifier

cap = cv2.VideoCapture(0)
time.sleep(1) # time for camera

delay = 1 # save image every 1 second
last_saved = time.time()
index = 1

# emoji stuff
emoji = cv2.imread('emojis/dizzy_face.png')

# load emotion classifier
c = classifier.load_classifier('model')

emoji_mappings = {'anger':'steaming', 'disgust':'dizzy_face', 'fear':'scream',
                  'happy':'triumph', 'sadness':'slight_frown', 'surprise':'flushed'}

window_name = "emojify"

def image_as_nparray(image):
    """
    Converts PIL's Image to numpy's array.
    :param image: PIL's Image object.
    :return: Numpy's array of the image.
    """
    return np.asarray(image)

def draw_with_alpha(source_image, image_to_draw, coordinates):
    x, y, w, h = coordinates
    image_to_draw = cv2.resize(image_to_draw, (h, w), Image.ANTIALIAS)
    s_img = image_to_draw
    l_img = source_image
    y_offset = y
    x_offset = x
    for y in range(s_img.shape[0]):
        for x in range(s_img.shape[1]):
            if not np.array_equal(s_img[y][x], emoji[0][0]):
                l_img[y_offset + y, x_offset + x] = s_img[y][x]

while(True):
    # Capture frame-by-frame
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        print("web cam not found")
        exit()
    while ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        emotions = classifier.extract_faces_and_emotions(c, gray)
        for (x,y,w,h), predicted_emotion in emotions:
            path_image_to_draw = 'emojis/' + emoji_mappings[predicted_emotion] + '.png'
            image_to_draw = cv2.imread(path_image_to_draw)
            draw_with_alpha(frame, image_to_draw, (x,y,w,h))

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
