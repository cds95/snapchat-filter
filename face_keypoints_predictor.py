import numpy as np
import cv2
from constants import IMG_SIZE
from keras.models import load_model


class FaceKeypointsPredictor:
    def __init__(self):
        self.model = load_model("./model/model.h5")

    def get_face_keypoints(self, img):
        face, x, y, w, h = self.extract_face(img)
        face = face / 255.0
        face = face.astype(np.float32)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = self.get_formatted_im(face)
        keypoint_pred = self.predict(face)
        return keypoint_pred, x, y, w, h

    # Note that image must be grayscale
    def extract_face(self, img):
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        face_cood = face_cascade.detectMultiScale(img, 1.3, 5)

        # Assume one face for now
        for (x, y, w, h) in face_cood:
            return img[y:y + h, x:x + w], x, y, w, h

    def predict(self, img):
        return self.model.predict(img)

    def get_formatted_im(self, img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if np.ndim(img) == 3 else img
        resized_img = cv2.resize(grayscale_img, (IMG_SIZE, IMG_SIZE))
        reshaped_img_for_prediction = self.expand_to_3_channels(resized_img)
        return np.resize(reshaped_img_for_prediction, (1, IMG_SIZE, IMG_SIZE, 3))

    def expand_to_3_channels(self, img):
        res = np.zeros((img.shape[0], img.shape[1], 3))
        res[:, :, 0] = img
        res[:, :, 1] = img
        res[:, :, 2] = img
        return res
