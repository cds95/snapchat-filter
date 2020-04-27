from face_keypoints_predictor import FaceKeypointsPredictor
import enum


class FilterType(enum.Enum):
    DOG = "DOG"
    RAINBOW = "RAINBOW"
    SUNGLASSES = "SUNGLASSES"


class BaseFilter:
    def __init__(self):
        self.facial_keypoints_predictor = FaceKeypointsPredictor()

    @staticmethod
    def apply_filter_type(img, filter_type):
        pass

    def apply(self, img):
        keypoint_pred, x, y, w, h = self.facial_keypoints_predictor.get_face_keypoints(img)
        keypoint_pred = keypoint_pred[0]
        face = img[y:y + h, x:x + w]
        return self.apply_filter(img, face, keypoint_pred, x, y, w, h)

    def apply_filter(self, img, face, keypoint_pred, face_x, face_y, face_width, face_height):
        pass
