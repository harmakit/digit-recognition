import cv2.cv2 as cv2


class ImageProcessor:

    @staticmethod
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def binarize(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    @staticmethod
    def filter(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def apply_morphology(image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def resize_and_crop(image):
        # crop to fit non-blank area
        non_blank = cv2.findNonZero(image)
        x, y, w, h = cv2.boundingRect(non_blank)
        image = image[y:y + h, x:x + w]
        # resize to 28x28
        return cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    @staticmethod
    def invert(image):
        return cv2.bitwise_not(image)

    def __process(self, image_path):
        image = cv2.imread(image_path)
        image = self.grayscale(image)
        image = self.binarize(image)
        image = self.filter(image)
        image = self.resize_and_crop(image)
        image = self.apply_morphology(image)
        image = self.resize_and_crop(image)
        image = self.invert(image)
        return image

    def get_image(self, image_path):
        image = self.__process(image_path)
        return image
