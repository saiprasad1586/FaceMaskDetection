from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import numpy as np


class Predict:
    def __init__(self):
        self.imgsize = (100, 100)
        self.model = load_model('../model.h5')

    def predictOnImg(self, img=None):
        # testimg = image.fromarray(img)
        testimg = cv2.resize(img, self.imgsize)
        testimg = testimg[..., ::-1]
        # testimg = image.load_img('Images/mask/1.jpg', target_size = self.imgsize)
        # testimg = Image.img_to_array(testimg)
        testimg = np.expand_dims(testimg,axis = 0)

        result = self.model.predict(testimg)
        # print(result)
        return int(result[0][0])


def main():
    predict = Predict()
    predict.predictOnImg()


if __name__ == '__main__':
    main()
