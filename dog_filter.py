from base_filter import BaseFilter
import cv2
import numpy as np
from pixel_point import PixelPoint

DOG_FILTER_X_OFFSET = 30 / 620
DOG_FILTER_Y_OFFSET = 414 / 425


class DogFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        dog_filter_path = "filter-images/dog-filter.png"
        dog_filter = cv2.imread(dog_filter_path)
        self.dog_filter = cv2.cvtColor(dog_filter, cv2.COLOR_BGR2RGB)

    def apply_filter(self, img, face, keypoint_pred, face_x, face_y, face_width, face_height):
        nose_x, nose_y = self.get_nose_row_col(keypoint_pred, face_width, face_height)
        left_eye_center_x, left_eye_center_y, right_eye_center_x, right_eye_center_y = self.get_eye_cood_for_dog(
            keypoint_pred, face)
        resized_dog_filter = self.get_resized_dog_filter(face, left_eye_center_x, right_eye_center_x, self.dog_filter)
        return self.overlay_dog_filter(resized_dog_filter, img, nose_x, nose_y, face_x, face_y, face_width, face_height)

    def get_nose_row_col(self, keypoint_pred, face_width, face_height):
        normalized_nose_x = keypoint_pred[20]
        normalized_nose_y = keypoint_pred[21]
        return PixelPoint.denormalize_point(normalized_nose_x, face_width), PixelPoint.denormalize_point(
            normalized_nose_y, face_height)

    def get_eye_cood_for_dog(self, keypoint_pred, face):
        face_height = face.shape[0]
        face_width = face.shape[1]
        normalized_left_eye_x = keypoint_pred[0]
        normalized_left_eye_y = keypoint_pred[1]
        normalized_right_eye_x = keypoint_pred[2]
        normalized_right_eye_y = keypoint_pred[3]

        left_eye_x = PixelPoint.denormalize_point(normalized_left_eye_x, face_width)
        left_eye_y = PixelPoint.denormalize_point(normalized_left_eye_y, face_height)
        right_eye_x = PixelPoint.denormalize_point(normalized_right_eye_x, face_width)
        right_eye_y = PixelPoint.denormalize_point(normalized_right_eye_y, face_height)
        return left_eye_x, left_eye_y, right_eye_x, right_eye_y

    def get_resized_dog_filter(self, face, left_eye, right_eye, dog_filter):
        face_height = face.shape[0]
        filter_width = 2 * abs(right_eye - left_eye)
        return cv2.resize(dog_filter, (int(filter_width), int(2 * face_height)))

    def overlay_dog_filter(self, dog_filter, img, nose_x, nose_y, x, y, face_width, face_height):
        res = np.copy(img)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        for row in range(0, dog_filter.shape[0]):
            for col in range(0, dog_filter.shape[1]):
                for channel in range(0, dog_filter.shape[2]):
                    if dog_filter[row][col][channel] > 0:
                        img_col = int(col + nose_x + x - face_width / 2 + dog_filter.shape[0] * DOG_FILTER_X_OFFSET)
                        img_row = int(row + nose_y + y - face_height / 2 - dog_filter.shape[1] * DOG_FILTER_Y_OFFSET)
                        res[img_row][img_col][channel] = dog_filter[row][col][channel]
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
