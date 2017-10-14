import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix

cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
image_size = 100
emotions_dictionary = [None, "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

def extract_face(image):
    x, y, w, h = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=6
    )[0]

    face = image[y:y + h, x:x + w]
    resized_face = cv2.resize(face, (image_size, image_size))

    return resized_face

def images_emotions_data(participant_id, session_id):
    emotion_files = glob.glob('CK+/Emotion/{}/{}/*'.format(participant_id, session_id))
    if len(emotion_files) == 0:
        return [[], []]

    emotion_filename = emotion_files[0]
    with open(emotion_filename, 'r') as emotion_file:
        emotion = emotions_dictionary[int(float(emotion_file.read()))]

    if emotion == 'contempt' or emotion is None: # remove contempt
        return [[], []]

    session_images = sorted(glob.glob('CK+/cohn-kanade-images/{}/{}/*'.format(participant_id, session_id)))

    length = int(len(session_images) / 4)

    images = [session_images[-1 - i] for i in range(length)]
    data = (extract_face(cv2.imread(image, cv2.IMREAD_GRAYSCALE)).flatten() for image in images)

    return [data, [emotion for i in range(length)]]

def generate_classifier():
    participants = glob.glob("CK+/Emotion/*")

    images = []
    emotions = []

    for participant in participants:
        # Get the participant ID from the last 4 chars
        participant_id = participant[-4:]

        sessions = glob.glob('{}/*'.format(participant))
        for session in sessions:
            # Get the participant ID from the last 3 chars
            session_id = session[-3:]

            current_image, current_emotion = images_emotions_data(participant_id, session_id)
            images += current_image
            emotions += current_emotion

    X, y = np.array(images), np.array(emotions)
    shuffle_index = np.random.permutation(len(X))
    X, y = X[shuffle_index], y[shuffle_index]

    clf = SGDClassifier(random_state=2017, max_iter=5)
    clf.fit(X, y)

    return clf

def predict_image(classifier, image):
    face = extract_face(image)
    return classifier.predict([face])
