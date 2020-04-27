from constants import IMG_SIZE

class PixelPoint:
    @staticmethod
    def denormalize_point(point, img_size=IMG_SIZE):
        return img_size * (point + 0.5)

    @staticmethod
    def normalize_point(point, img_size=IMG_SIZE):
        return point / img_size - 0.5