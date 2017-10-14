import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split

cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
image_size = 100
emotions_dictionary = [None, 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

def extract_face(image):

    face_result = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=6
    )

    if len(face_result) == 0:
        return []

    x, y, w, h = face_result[0]
    # x, y, w, h = face_cascade.detectMultiScale(
    #     image,
    #     scaleFactor=1.1,
    #     minNeighbors=6
    # )[0]

    face = image[y:y + h, x:x + w]
    resized_face = cv2.resize(face, (image_size, image_size))

    return resized_face

def load_and_process_image(filename):
    result = extract_face(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    if len(result) == 0:
        return result
    else:
        return result.flatten()
    # return extract_face(cv2.imread(filename, cv2.IMREAD_GRAYSCALE)).flatten()

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
    data = (load_and_process_image(image_file) for image_file in images)

    return [data, [emotion for i in range(length)]]

def generate_classifier():
    participants = glob.glob('CK+/Emotion/*')

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

def save_classifier(classifier, filename):
    with open(filename, 'sr+') as f:
        p = pickle.dumps(classifier)
        f.write(p)

def load_classifier(filename):
    with open(filename, 'rb') as f:
        return pickle.loads(f.read())

def predict_image(classifier, filename):
    face = load_and_process_image(filename)
    if len(face) == 0:
        print("no face detected")
        return -1
    return classifier.predict([face])[0]
