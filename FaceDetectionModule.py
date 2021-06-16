import cv2
import mediapipe as mp
import time
from CNNPredictor.CnnPrediction import Predict


class FaceDetector:

    def __init__(self, minDetectionCon=0.75):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.results = ['Mask', 'No Mask']
        self.predict = Predict()

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        # count = 0
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox
                cropped_img = img[y:y + h, x:x + w]
                result = self.predict.predictOnImg(cropped_img)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    if result != None:
                        if result == 1:
                            x = 'No Mask'
                            y = 0
                        elif result == 0:
                            x = 'Mask'
                            y = 1
                        img = self.rectdraw(img, bbox, y)
                        cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                    (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 255), 2)
                        if y == 1:
                            cv2.putText(img, f'{x}',
                                        (bbox[0] + 80, bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                        2, (255, 0, 255), 2)
                        elif y == 0:
                            cv2.putText(img, f'{x}',
                                        (bbox[0] + 80, bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                        2, (0, 0, 255), 2)
        return img, bboxs

    def rectdraw(self, img, bbox, mask, l=30, t=5, rt=1, ):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        if mask == 1:
            cv2.rectangle(img, bbox, (255, 0, 255), rt)
            # Top Left  x,y
            cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
            cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
            # Top Right  x1,y
            cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
            cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
            # Bottom Left  x,y1
            cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
            cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
            # Bottom Right  x1,y1
            cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
            cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        elif mask == 0:
            cv2.rectangle(img, bbox, (0, 0, 255), rt)
            # Top Left  x,y
            cv2.line(img, (x, y), (x + l, y), (0, 0, 255), t)
            cv2.line(img, (x, y), (x, y + l), (0, 0, 255), t)
            # Top Right  x1,y
            cv2.line(img, (x1, y), (x1 - l, y), (0, 0, 255), t)
            cv2.line(img, (x1, y), (x1, y + l), (0, 0, 255), t)
            # Bottom Left  x,y1
            cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
            cv2.line(img, (x, y1), (x, y1 - l), (0, 0, 255), t)
            # Bottom Right  x1,y1
            cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
            cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)

        return img

    def cropImage(self, img, bbox, mask, count):
        x, y, w, h = bbox
        croped_img = img[y:y + h, x:x + w]
        if mask:
            cv2.imwrite('Images/mask/{}.jpg'.format(count), croped_img)
        else:
            cv2.imwrite('Images/no_mask/{}.jpg'.format(count), croped_img)


def main():
    cap = cv2.VideoCapture(0)
    f = 0
    detector = FaceDetector()
    count = 0
    count1 = 400
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        count1 += 1
        count += 5
        i = time.time()
        fps = 1 / (i - f)
        f = i
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
