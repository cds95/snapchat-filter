from base_filter import BaseFilter
import cv2
import numpy as np
from pixel_point import PixelPoint

RAINBOW_FILTER_SCALE = 4  # Mouth only take up about a quarter of the image
RAINBOW_X_OFFSET = 0 / 400
RAINBOW_Y_OFFSET = 0 / 400
RAINBOW_HEIGHT = 130


class RainbowFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        rainbow_filter_path = "filter-images/rainbow-mouth.png"
        rainbow = cv2.imread(rainbow_filter_path)
        self.rainbow = rainbow.astype(np.uint8)

    def apply_filter(self, img, face, keypoint_pred, face_x, face_y, face_width, face_height):
        mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y, lip_top_x, lip_top_y = self.get_mouth(face,
                                                                                                        keypoint_pred)
        resized_rainbow = self.resize_rainbow(self.rainbow, mouth_left_x, mouth_right_x)
        return self.overlay_rainbow(resized_rainbow, img, face_x, face_y, mouth_right_x, lip_top_y)

    def get_mouth(self, face, keypoint_pred):
        face_height = face.shape[0]
        face_width = face.shape[1]

        norm_mouth_left_x = keypoint_pred[22]
        norm_mouth_left_y = keypoint_pred[23]
        norm_mouth_right_x = keypoint_pred[24]
        norm_mouth_right_y = keypoint_pred[25]
        norm_lip_top_x = keypoint_pred[26]
        norm_lip_top_y = keypoint_pred[27]

        mouth_left_x = PixelPoint.denormalize_point(norm_mouth_left_x, face_width)
        mouth_left_y = PixelPoint.denormalize_point(norm_mouth_left_y, face_height)
        mouth_right_x = PixelPoint.denormalize_point(norm_mouth_right_x, face_width)
        mouth_right_y = PixelPoint.denormalize_point(norm_mouth_right_y, face_height)
        lip_top_x = PixelPoint.denormalize_point(norm_lip_top_x, face_width)
        lip_top_y = PixelPoint.denormalize_point(norm_lip_top_y, face_height)
        return mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y, lip_top_x, lip_top_y

    def resize_rainbow(self, rainbow_filter, mouth_left_x, mouth_right_x):
        width = abs(mouth_left_x - mouth_right_x) * RAINBOW_FILTER_SCALE
        width = int(width)
        return cv2.resize(rainbow_filter, (width, RAINBOW_HEIGHT))

    def overlay_rainbow(self, rainbow_filter, img, face_x, face_y, mouth_right_x, lip_top_y):
        res = np.copy(img)
        for row in range(0, rainbow_filter.shape[0]):
            for col in range(0, rainbow_filter.shape[1]):
                for channel in range(0, rainbow_filter.shape[2]):
                    if rainbow_filter[row][col][channel] > 0:
                        img_col = int(col + mouth_right_x + face_x) - int(rainbow_filter.shape[0] * RAINBOW_X_OFFSET) - 120
                        img_row = int(row + lip_top_y + face_y) - int(rainbow_filter.shape[1] * RAINBOW_Y_OFFSET)
                        res[img_row][img_col][channel] = rainbow_filter[row][col][channel]
        return res
