# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.wgt_video = QVideoWidget(self.centralwidget)  # 创建一个 视频播放组件
        self.wgt_video.setGeometry(QtCore.QRect(50, 20, 500, 300))
        self.wgt_video.setObjectName("wgt_video")

        self.btn_open = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open.setGeometry(QtCore.QRect(260, 320, 120, 50))
        self.btn_open.setObjectName("btn_open")
        MainWindow.setCentralWidget(self.centralwidget)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_open.setText(_translate("MainWindow", "打开视频文件并播放"))

        self.player = QMediaPlayer()  # 创建视频播放管理器
        self.player.setVideoOutput(self.wgt_video)  # 视频播放输出的widget，就是上面定义的
        self.player.set
        self.btn_open.clicked.connect(self.openVideoFile)   # 打开视频文件按钮

    def openVideoFile(self):
        print('open file')
         # 播放网络视频
        # self.player.setMedia(QMediaContent(QUrl.fromLocalFile("https://vd1.bdstatic.com/mda-hg6uempmez9u6mqi/sc/mda-hg6uempmez9u6mqi.mp4?auth_key=1562172911-0-0-4c22196ad1d0fcc49402d91336c999c5&bcevod_channel=searchbox_feed&pd=bjh&abtest=all")))
        #选择本地视频播放
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取本地视频文件
        self.player.play()  # 播放视频

# if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # vieo_gui = Ui_MainWindow()
    # vieo_gui.show()
    # sys.exit(app.exec_())


import numpy as np
from collections import defaultdict

ninf = float("-inf")


# 求每一列(即每个时刻)中最大值对应的softmax值
def softmax(logits):
	# 注意这里求e的次方时，次方数减去max_value其实不影响结果，因为最后可以化简成教科书上softmax的定义
	# 次方数加入减max_value是因为e的x次方与x的极限(x趋于无穷)为无穷，很容易溢出，所以为了计算时不溢出，就加入减max_value项
	# 次方数减去max_value后，e的该次方数总是在0到1范围内。
	max_value = np.max(logits, axis=1, keepdims=True)
	exp = np.exp(logits - max_value)
	exp_sum = np.sum(exp, axis=1, keepdims=True)
	dist = exp / exp_sum
	return dist


def remove_blank(labels, blank=0):
	new_labels = []
	# 合并相同的标签
	previous = None
	for l in labels:
		if l != previous:
			new_labels.append(l)
			previous = l
	# 删除blank
	new_labels = [l for l in new_labels if l != blank]

	return new_labels


def insert_blank(labels, blank=0):
	new_labels = [blank]
	for l in labels:
		new_labels += [l, blank]
	return new_labels


def _logsumexp(a, b):
	'''
	np.log(np.exp(a) + np.exp(b))

	'''

	if a < b:
		a, b = b, a

	if b == ninf:
		return a
	else:
		return a + np.log(1 + np.exp(b - a))


def logsumexp(*args):
	'''
	from scipy.special import logsumexp
	logsumexp(args)
	'''
	res = args[0]
	for e in args[1:]:
		res = _logsumexp(res, e)
	return res


def prefix_beam_decode(y, beam_size=10, blank=0):
	T, V = y.shape
	log_y = np.log(y)
	# 最后一个字符是blank与最后一个字符为non-blank两种情况
	beam = [(tuple(), (0, ninf))]
	# 对于每一个时刻t
	for t in range(T):
		# 当我使用普通的字典时，用法一般是dict={},添加元素的只需要dict[element] =value即可，调用的时候也是如此
		# dict[element] = xxx,但前提是element字典里，如果不在字典里就会报错
		# defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
		# dict =defaultdict( factory_function)
		# 这个factory_function可以是list、set、str等等，作用是当key不存在时，返回的是工厂函数的默认值
		# 这里就是(ninf, ninf)是默认值
		new_beam = defaultdict(lambda: (ninf, ninf))
		# 对于beam中的每一项
		for prefix, (p_b, p_nb) in beam:
			for i in range(V):
				# beam的每一项都加上时刻t中的每一项
				p = log_y[t, i]
				# 如果i中的这项是blank
				if i == blank:
					# 将这项直接加入路径中
					new_p_b, new_p_nb = new_beam[prefix]
					new_p_b = logsumexp(new_p_b, p_b + p, p_nb + p)
					new_beam[prefix] = (new_p_b, new_p_nb)
					continue
				# 如果i中的这一项不是blank
				else:
					end_t = prefix[-1] if prefix else None
					# 判断之前beam项中的最后一个元素和i的元素是不是一样
					new_prefix = prefix + (i,)
					new_p_b, new_p_nb = new_beam[new_prefix]
					# 如果不一样，则将i这项加入路径中
					if i != end_t:
						new_p_nb = logsumexp(new_p_nb, p_b + p, p_nb + p)
					else:
						new_p_nb = logsumexp(new_p_nb, p_b + p)
					new_beam[new_prefix] = (new_p_b, new_p_nb)
					# 如果一样，保留现有的路径，但是概率上要加上新的这个i项的概率
					if i == end_t:
						new_p_b, new_p_nb = new_beam[prefix]
						new_p_nb = logsumexp(new_p_nb, p_nb + p)
						new_beam[prefix] = (new_p_b, new_p_nb)

		# 给新的beam排序并取前beam_size个
		beam = sorted(new_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
		beam = beam[:beam_size]

	return beam


np.random.seed(1111)
y_test = softmax(np.random.random([20, 6]))
beam_test = prefix_beam_decode(y_test, beam_size=100)
for beam_string, beam_score in beam_test[:20]:
	print(remove_blank(beam_string), beam_score)






