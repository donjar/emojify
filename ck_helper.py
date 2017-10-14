import glob
import cv2

# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
participants = glob.glob("CK+/Emotion/*")

images = []
emotions = []

for participant in participants:
    # Get the participant ID from the last 4 chars
    participant_id = participant[-4:]

    sessions = glob.glob('{}/*'.format(participant))
    for session in sessions:
        # Get the participant ID from the last 4 chars
        session_id = session[-3:]

        emotion_files = glob.glob('{}/*'.format(session))
        if len(emotion_files) == 0:
            continue

        emotion_file = emotion_files[0]
        emotion = int(float(open(emotion_file, 'r').read()))

        emotions.append(0) # neutral face
        emotions.append(emotion)

        session_images = sorted(glob.glob('CK+/cohn-kanade-images/{}/{}/*'.format(participant_id, session_id)))
        neutral_image = session_images[0]
        emotion_image = session_images[-1]

        neutral_data = cv2.imread(neutral_image, cv2.IMREAD_GRAYSCALE).flatten()
        emotion_data = cv2.imread(emotion_image, cv2.IMREAD_GRAYSCALE).flatten()

        images.append(neutral_data)
        images.append(emotion_data)
