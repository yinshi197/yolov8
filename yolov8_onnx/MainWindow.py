from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QFont 
from ultralytics import YOLO # type: ignore
import cv2
import torch
import numpy as np

class ImagePopup(QtWidgets.QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.initUI()

    def initUI(self):
        # 固定窗口大小为 1920x1080
        self.setFixedSize(1600, 900)

        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap.fromImage(self.image)
        # 调整图片大小以适应窗口
        scaled_pixmap = pixmap.scaled(1600, 900, 
                                      QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                      transformMode=QtCore.Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)
        layout.addWidget(self.label)
        self.setWindowTitle("放大图片")

class VideoThread(QtCore.QThread):
    updateFrame = QtCore.pyqtSignal(QtGui.QImage)
    results = QtCore.pyqtSignal(list)
    
    def __init__(self, video_file, model, classIndexes, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.model = model
        self.running = False
        self.parent = parent
        self.classIndexes = classIndexes
        self.cap = None

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_file)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img = frame
            results = self.model(img, stream=True, classes=self.classIndexes)
            results = list(results)
            annotated_img = results[0].plot()
            qimg = QtGui.QImage(annotated_img.data, annotated_img.shape[1], annotated_img.shape[0], annotated_img.strides[0], QtGui.QImage.Format.Format_BGR888)
            self.updateFrame.emit(qimg)
            self.results.emit(results)
        cap.release()
        self.running = False

    def stop(self):
        self.running = False
    
    def continue_video(self):
        self.running = True
        self.run()
        
    def setClassIndexes(self, classIndexes):
        self.classIndexes = classIndexes

class Ui_MainWindow(QtWidgets.QWidget):
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(931, 592)
        
        self.videoThread = None

        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # QLabel for Video/Image Display
        self.videoLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.videoLabel.setGeometry(QtCore.QRect(30, 70, 531, 291))
        self.videoLabel.setObjectName("videoLabel")

        # 设置封面图片
        self.setCoverImage()

        # Text Browser
        self.textBrowser = QtWidgets.QTextBrowser(parent=self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(590, 70, 321, 291))
        self.textBrowser.setObjectName("textBrowser")

        # Label for Behavior Statistics
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(590, 48, 120, 20))
        self.label_2.setObjectName("label_2")
        self.label_2.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold))

        # Label for Monitoring Results
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 50, 70, 20))
        self.label_3.setObjectName("label_3")
        self.label_3.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold))

        # Main Title Label
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(290, 20, 250, 20))
        self.label_4.setObjectName("label_4")
        self.label_4.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Weight.Bold))
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Control Widget
        self.widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 380, 870, 170))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        # Horizontal Layout for Checkboxes
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(parent=self.widget)
        self.label.setObjectName("label")
        self.label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold))
        self.horizontalLayout.addWidget(self.label)
        self.checkBox = QtWidgets.QCheckBox(parent=self.widget)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout.addWidget(self.checkBox)
        self.checkBox_2 = QtWidgets.QCheckBox(parent=self.widget)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout.addWidget(self.checkBox_2)
        self.checkBox_3 = QtWidgets.QCheckBox(parent=self.widget)
        self.checkBox_3.setObjectName("checkBox_3")
        self.horizontalLayout.addWidget(self.checkBox_3)
        self.checkBox_4 = QtWidgets.QCheckBox(parent=self.widget)
        self.checkBox_4.setObjectName("checkBox_4")
        self.horizontalLayout.addWidget(self.checkBox_4)
        self.checkBox_5 = QtWidgets.QCheckBox(parent=self.widget)
        self.checkBox_5.setObjectName("checkBox_5")
        self.horizontalLayout.addWidget(self.checkBox_5)
        self.checkBox_6 = QtWidgets.QCheckBox(parent=self.widget)
        self.checkBox_6.setObjectName("checkBox_6")
        self.horizontalLayout.addWidget(self.checkBox_6)
        self.verticalLayout.addLayout(self.horizontalLayout)

        # Horizontal Layout for Buttons
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        
        self.pushButton = QtWidgets.QPushButton(parent=self.widget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        
        self.pushButton_2 = QtWidgets.QPushButton(parent=self.widget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        #第二层按钮布局
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        
        # 添加放大图片按钮
        self.zoomButton = QtWidgets.QPushButton(parent=self.widget)
        self.zoomButton.setObjectName("zoomButton")
        self.horizontalLayout_3.addWidget(self.zoomButton)
        
        # 添加停止监控按钮
        self.stopButton = QtWidgets.QPushButton(parent=self.widget)
        self.stopButton.setObjectName('stopButton')
        self.horizontalLayout_3.addWidget(self.stopButton)

        # 添加继续监控按钮
        self.continueButton = QtWidgets.QPushButton(parent=self.widget)
        self.continueButton.setObjectName('continueButton')
        self.horizontalLayout_3.addWidget(self.continueButton)
        
        self.verticalLayout.addLayout(self.horizontalLayout_3)


        # Set Central Widget
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 931, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        # 连接按钮点击事件
        self.pushButton.clicked.connect(self.openImageFile)
        self.pushButton_2.clicked.connect(self.openVideoFile)
        self.zoomButton.clicked.connect(self.showZoomedImage)
        self.stopButton.clicked.connect(self.stopMonitoring)
        self.continueButton.clicked.connect(self.continueMonitoring)

        self.checkBox.stateChanged.connect(self.updateCheckBoxState)
        self.checkBox_2.stateChanged.connect(self.updateCheckBoxState)
        self.checkBox_3.stateChanged.connect(self.updateCheckBoxState)
        self.checkBox_4.stateChanged.connect(self.updateCheckBoxState)
        self.checkBox_5.stateChanged.connect(self.updateCheckBoxState)
        self.checkBox_6.stateChanged.connect(self.updateCheckBoxState)

        # 加载模型
        self.model = YOLO("models/best_last.pt")
        
    def updateCheckBoxState(self):
        if self.videoThread is not None:
            self.videoThread.setClassIndexes(self.SelectClass())
    
    def setCoverImage(self):
        """设置封面图片"""
        cover_image_path = 'cover3.jpg'  # 图片路径
        pixmap = QtGui.QPixmap(cover_image_path)
        self.videoLabel.setPixmap(pixmap.scaled(
            self.videoLabel.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            transformMode=QtCore.Qt.TransformationMode.SmoothTransformation
        ))
        
    def preprocessImage(self, image_path):
        """预处理图像"""
        img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def detectObjects(self, img):
        """使用模型进行检测"""
        classIndex = self.SelectClass()
        results = self.model(img, stream=True, classes=classIndex, show_labels=True)
        results = list(results)  # 将迭代器转换为列表
        self.AnalyzeResults(results)
        return results
    
    def AnalyzeResults(self, results):
        """分析检测结果"""
        class_counts = {}
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                if cls in class_counts:
                    class_counts[cls] += 1
                else:
                    class_counts[cls] = 1
        self.update_info(class_counts, results)
    
    def update_info(self, class_counts, results):
        """更新类别统计信息"""
        self.textBrowser.clear()
        classnames = ['举手', '看书', '写字', '使用手机', '低头做其他事', '睡觉']
        info_text = "--------分析结果仅供参考--------\n"

        # 初始化统计数据
        total_confidence = 0.0  # 置信度之和
        avg_confidences = {}

        # 计算每个类别的平均置信度
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                if cls in class_counts:
                    if cls in avg_confidences:
                        avg_confidences[cls]['total_conf'] += conf
                        avg_confidences[cls]['count'] += 1
                    else:
                        avg_confidences[cls] = {'total_conf': conf, 'count': 1}
                    total_confidence += conf  # 累加置信度

        # 输出统计信息
        for cls, count in class_counts.items():
            avg_confidence = 0
            if cls in avg_confidences and avg_confidences[cls]['count'] > 0:
                avg_confidence = avg_confidences[cls]['total_conf'] / avg_confidences[cls]['count']
            info_text += f"{classnames[cls]}人数: {count}, 平均置信度: {avg_confidence:.2f}\n"

        # 添加总人数和总置信度
        info_text += f"\n总人数: {sum(class_counts.values())}, 总置信度: {total_confidence:.2f}\n"

        # 设置字体大小
        font = QFont()
        font.setPointSize(12)  # 设置字体大小为12
        self.textBrowser.setFont(font)
        
        # 更新文本显示
        self.textBrowser.setText(info_text)
    def SelectClass(self):
        SelectClassIndex = []
        if(self.checkBox.isChecked()):
            SelectClassIndex.append(0)
        
        if(self.checkBox_2.isChecked()):
            SelectClassIndex.append(1)
        
        if(self.checkBox_3.isChecked()):
            SelectClassIndex.append(2)
        
        if(self.checkBox_4.isChecked()):
            SelectClassIndex.append(3)
        if(self.checkBox_5.isChecked()):
            SelectClassIndex.append(4)
        if(self.checkBox_6.isChecked()):
            SelectClassIndex.append(5)
        
        return SelectClassIndex
        
    def openImageFile(self):
        """打开图片文件对话框"""
        if not any([self.checkBox.isChecked(), self.checkBox_2.isChecked(), self.checkBox_3.isChecked(), self.checkBox_4.isChecked(), self.checkBox_5.isChecked(), self.checkBox_6.isChecked()]):
            QtWidgets.QMessageBox.warning(self, "警告", "请选择至少一种行为进行监测！")
            return

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "选择图片文件", "", "图片文件 (*.png *.jpg *.jpeg)")
        if file_name:
            img = self.preprocessImage(file_name)
            results = self.detectObjects(img)
            annotated_img = results[0].plot()  # 获取带有标注的图片
            pixmap = self.cvMatToQPixmap(annotated_img)
            self.videoLabel.setPixmap(pixmap.scaled(
                self.videoLabel.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=QtCore.Qt.TransformationMode.SmoothTransformation
            ))
            self.detected_image = annotated_img  # 保存检测后的图片
            print(f"选择了图片文件: {file_name}")
            self.zoomButton.setDisabled(False)
            self.continueButton.setDisabled(True)

    def showZoomedImage(self):
        """显示放大图片的弹出窗口"""
        if self.detected_image is not None:
            qimg = QtGui.QImage(self.detected_image.data, self.detected_image.shape[1], self.detected_image.shape[0], self.detected_image.strides[0], QtGui.QImage.Format.Format_BGR888)
            popup = ImagePopup(qimg)
            popup.exec()
    def openVideoFile(self):
        if not any([self.checkBox.isChecked(), self.checkBox_2.isChecked(), self.checkBox_3.isChecked(), self.checkBox_4.isChecked(), self.checkBox_5.isChecked(), self.checkBox_6.isChecked()]):
            QtWidgets.QMessageBox.warning(self, "警告", "请选择至少一种行为进行监测！")
            return
        
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if file_name:
            if self.videoThread is not None:
                self.videoThread.stop()
                self.videoThread.wait()
            
            classIndex = self.SelectClass()
            self.videoThread = VideoThread(file_name, self.model, classIndex, MainWindow)
            self.videoThread.updateFrame.connect(self.updateVideoFrame)
            self.videoThread.results.connect(self.updateInfo)
            self.videoThread.start()
            print(f"选择了视频文件: {file_name}")
            self.stopButton.setDisabled(False)
            
            self.pushButton.setDisabled(True)
            self.pushButton_2.setDisabled(True)
            self.continueButton.setDisabled(True)
            self.zoomButton.setDisabled(True)
    
    def stopMonitoring(self):
        """停止视频监控"""
        if self.videoThread is not None:
            self.videoThread.stop()
            #self.videoThread.wait()
            #self.videoThread = None
            self.stopButton.setDisabled(True)
            self.continueButton.setDisabled(False)

            self.pushButton.setDisabled(False)
            self.pushButton_2.setDisabled(False)
    
    def continueMonitoring(self):
        """继续视频监控"""
        if self.videoThread is not None:
            self.videoThread.start()
            self.continueButton.setDisabled(True)
            self.stopButton.setDisabled(False)

            self.pushButton.setDisabled(True)
            self.pushButton_2.setDisabled(True)

    def updateVideoFrame(self, qimg):
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pixmap.scaled(
            self.videoLabel.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            transformMode=QtCore.Qt.TransformationMode.SmoothTransformation
        ))
    
    def updateInfo(self, results):
        self.AnalyzeResults(results)

    def cvMatToQPixmap(self, img):
        """将OpenCV图像转换为QPixmap"""
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimg = QtGui.QImage(img.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        return pixmap

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "课堂行为监测分析系统"))
        self.label_2.setText(_translate("MainWindow", "行为统计与分析"))
        self.label_3.setText(_translate("MainWindow", "监测结果"))
        self.label_4.setText(_translate("MainWindow", "课堂行为监测分析系统"))
        self.label.setText(_translate("MainWindow", "行为监测："))
        
        self.checkBox.setText(_translate("MainWindow", "举手"))
        self.checkBox_2.setText(_translate("MainWindow", "看书"))
        self.checkBox_3.setText(_translate("MainWindow", "写字"))
        self.checkBox_4.setText(_translate("MainWindow", "使用手机"))
        self.checkBox_5.setText(_translate("MainWindow", "低头做其他事"))
        self.checkBox_6.setText(_translate("MainWindow", "睡觉"))
        
        self.pushButton.setText(_translate("MainWindow", "图片监测"))
        self.pushButton_2.setText(_translate("MainWindow", "视频监测"))
        self.zoomButton.setText(_translate("MainWindow", "放大图片"))
        self.stopButton.setText(_translate("MainWindow", "停止监测"))
        self.continueButton.setText(_translate("MainWindow", "继续监测"))

# Example usage:
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.zoomButton.setDisabled(True)
    ui.stopButton.setDisabled(True)
    ui.continueButton.setDisabled(True)
    MainWindow.show()
    sys.exit(app.exec())