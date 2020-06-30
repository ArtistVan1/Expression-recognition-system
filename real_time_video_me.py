import cv2
import imutils
import numpy as np
from PyQt5 import QtGui, QtWidgets
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from load_and_process import preprocess_input


class Emotion_Rec:
    def __init__(self, model_path=None):


        detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

        if model_path == None:
            emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
        else:
            emotion_model_path = model_path


        self.face_detection = cv2.CascadeClassifier(detection_model_path)  # 级联分类器


        self.emotion_classifier = load_model(emotion_model_path, compile=False)

        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]

    def run(self, frame_in, canvas, label_face, label_result):
        # frame_in 摄像画面或图像
        # canvas 用于显示的背景图
        # label_face 用于人脸显示画面的label对象
        # label_result 用于显示结果的label对象

        # 调节画面大小
        frame = imutils.resize(frame_in, width=300)  # 缩放画面
        # frame = cv2.resize(frame, (300,300))  # 缩放画面
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图

        # 检测人脸
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        preds = []  # 预测的结果
        label = None  # 预测的标签
        (fX, fY, fW, fH) = None, None, None, None  # 人脸位置
        frameClone = frame.copy()  # 复制画面

        if len(faces) > 0:
            faces = sorted(faces, reverse=False, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))  # 按面积从小到大排序

            for i in range(len(faces)):
                # if i == 0:
                #     i = -1
                # else:
                #     break

                (fX, fY, fW, fH) = faces[i]
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, self.emotion_classifier.input_shape[1:3])
                roi = preprocess_input(roi)
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.emotion_classifier.predict(roi)[0]
                # emotion_probability = np.max(preds)  #
                label = self.EMOTIONS[preds.argmax()]  #

                cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0), 1)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 255, 0), 1)

        # canvas = 255* np.ones((250, 300, 3), dtype="uint8")
        # canvas = cv2.imread('slice.png', flags=cv2.IMREAD_UNCHANGED)

        for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
            # 用于显示各类别概率
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300) + 7
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (224, 200, 130), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)


        frameClone = cv2.resize(frameClone, (420, 280))


        show = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QtWidgets.QApplication.processEvents()

        show = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_result.setPixmap(QtGui.QPixmap.fromImage(showImage))

        return (label)
