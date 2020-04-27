from base_filter import BaseFilter
import cv2
import numpy as np
from pixel_point import PixelPoint

SUNGLASSES_X_OFFSET = 1000 / 3300
SUNGLASSES_Y_OFFSET = 250 / 1500
FILTER_SCALE = 1.5


class SunglassesFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        sunglasses_filter_path = "filter-images/frames.png"
        sunglasses = cv2.imread(sunglasses_filter_path)
        self.sunglasses = sunglasses.astype(np.uint8)

    def apply_filter(self, img, face, keypoint_pred, face_x, face_y, face_width, face_height):
        left_eye_x, left_eye_y, right_eye_x, right_eye_y = self.get_eye_cood_for_sunglasses(face, keypoint_pred)
        resized_sunglasses = self.resize_sunglasses(self.sunglasses, left_eye_x, right_eye_x)
        return self.overlay_sunglasses(resized_sunglasses, img, face_x, face_y, right_eye_x, right_eye_y)

    def get_eye_cood_for_sunglasses(self, face, keypoint_pred):
        face_height = face.shape[0]
        face_width = face.shape[1]
        normalized_left_eye_x = keypoint_pred[6]
        normalized_left_eye_y = keypoint_pred[7]
        normalized_right_eye_x = keypoint_pred[10]
        normalized_right_eye_y = keypoint_pred[11]
        left_eye_x = PixelPoint.denormalize_point(normalized_left_eye_x, face_width)
        left_eye_y = PixelPoint.denormalize_point(normalized_left_eye_y, face_height)
        right_eye_x = PixelPoint.denormalize_point(normalized_right_eye_x, face_width)
        right_eye_y = PixelPoint.denormalize_point(normalized_right_eye_y, face_height)
        return left_eye_x, left_eye_y, right_eye_x, right_eye_y

    def resize_sunglasses(self, sunglasses, left_eye_x, right_eye_x):
        width = abs(right_eye_x - left_eye_x) * FILTER_SCALE
        height = sunglasses.shape[0] * width / sunglasses.shape[1]  # To keep ratio
        width = int(width)
        height = int(height)
        return cv2.resize(sunglasses, (width, height))

    def overlay_sunglasses(self, sunglasses_filter, img, face_x, face_y, right_eye_x, right_eye_y):
        res = np.copy(img)
        for row in range(0, sunglasses_filter.shape[0]):
            for col in range(0, sunglasses_filter.shape[1]):
                for channel in range(0, sunglasses_filter.shape[2]):
                    if sunglasses_filter[row][col][channel] < 255:
                        img_col = int(col + right_eye_x + face_x) - int(
                            sunglasses_filter.shape[0] * SUNGLASSES_X_OFFSET)
                        img_row = int(row + right_eye_y + face_y) - int(
                            sunglasses_filter.shape[1] * SUNGLASSES_Y_OFFSET)
                        res[img_row][img_col][channel] = sunglasses_filter[row][col][channel]
        return res
