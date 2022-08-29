import os
import re

from PySide2 import QtWidgets
from PySide2.QtWidgets import *
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from analysis_back import *
from parameters import Parameters
from image import ImageInform, Image

__author__ = "{{Yuliang_Liu}} ({{s4564914}}), {{Wencan Peng}} ({{46222378}})"
__email__ = "yuliang.liu@uqconnect.edu.au, wencan.peng@uqconnect.edu.au"
__date__ = "16/08/2022"
__version__ = "2.0"


# ------------------ MplWidget ------------------
class MplWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(NavigationToolbar(self.canvas, self))

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)


# ------------------ MainWidget ------------------
class MainWidget(QWidget):

    def __init__(self):
        # open ui
        QWidget.__init__(self)
        # the name of ui file
        designer_file = QFile("wormanalysis.ui")
        designer_file.open(QFile.ReadOnly)
        # load ui
        loader = QUiLoader()
        loader.registerCustomWidget(MplWidget)
        self.ui = loader.load(designer_file, self)

        designer_file.close()

        self.image_name = ''
        self.image_path = ''
        self.cwd = os.getcwd()
        self.save_path = os.path.join(self.cwd, '1.csv')

        self.position_file = ''
        self.x = []
        self.y = []
        self.ui.text_file_path_2.setText(self.save_path)

        self.image_num = 0
        # self.image_8bit = None
        # self.image_16bit = None
        # self.image_bright = None

        # self.stop = False
        self.flip = False

        # instantiate and initialize Parameters class (with default values)
        # 实例化初始Parameters类（有默认值）
        self.parameters = Parameters()
        # show the parameter values in UI
        # 在GUI界面显示参数值
        self.initialization_parameter()

        self.img_inform: ImageInform = None  # it should be ImageInform class
        self.results = []

        self.image_nums = []
        self.right_brightness = []
        self.left_brightness = []
        self.brightness = []

        # self.rows = []
        # self.columns = []
        # 实例化后端线程类/instantiate QThread class and start back-end
        self.thread = QThread()
        self.i_thread = ImageProcessingThread()
        # moveToThread方法把实例化线程移到Thread管理
        self.i_thread.moveToThread(self.thread)
        # 连接槽函数
        self.i_thread.start_image_process_thread_signal.connect(
            self.i_thread.image_processing)
        self.i_thread.show_img_signal.connect(self.show_image)
        self.i_thread.show_img_signal_loop.connect(self.show_image_loop)
        self.i_thread.loop_signal.connect(self.i_thread.loop)
        # 开启线程,一直挂在后台
        self.thread.start()

        # 部件处理
        self.ui.button_select_file.clicked.connect(self.button_select_file)
        self.ui.button_save_data.clicked.connect(self.button_save_data)
        self.ui.button_position.clicked.connect(self.open_position_file)

        self.ui.button_next.clicked.connect(self.button_next)
        self.ui.button_last.clicked.connect(self.button_last)

        self.ui.button_go.clicked.connect(self.button_go)
        self.ui.button_run.clicked.connect(self.button_run)
        self.ui.button_refresh.clicked.connect(self.button_refresh)

        self.ui.button_pause.clicked.connect(self.button_pause)
        self.ui.button_stop.clicked.connect(self.button_kill)

        self.ui.button_up.clicked.connect(lambda: self.button_refresh(-1, 0))
        self.ui.button_down.clicked.connect(lambda: self.button_refresh(1, 0))
        self.ui.button_left.clicked.connect(lambda: self.button_refresh(0, -1))
        self.ui.button_right.clicked.connect(lambda: self.button_refresh(0, 1))

        self.ui.checkbox_mirror_symmetry.stateChanged.connect(
            self.checkbox_mirror_symmetry)

        self.ui.text_file_path.setText(self.image_path)
        self.dataframe = pd.DataFrame(
            columns=['Right_row', 'Right_column', 'Right_brightness',
                     'Left_row', 'Left_column', 'Left_brightness',
                     'Brightness'])
        self.ui.box_neuron_amount.textChanged.connect(self.set_neuron_amount)

        # widgets: swap neuron positions
        self.ui.button_swap.clicked.connect(self.swap)

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self,
                                               'Quit',
                                               "Quit?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            os._exit(0)
        else:
            event.ignore()

    def button_save_data(self):
        self.save_path, _ = QFileDialog.getSaveFileName(self,
                                                        'Select save path',
                                                        self.cwd,
                                                        'Table(*.csv)')
        self.ui.text_file_path_2.setText(self.save_path)

    def button_select_file(self):
        image_path_name, _ = QFileDialog.getOpenFileName(self, 'Select image',
                                                         '',
                                                         'Image files(*.tif)')
        self.image_path, self.image_name = os.path.split(image_path_name)
        self.ui.text_file_path.setText(self.image_path)

        regex = re.compile(r'\d+')
        if bool(re.search(r'\d', self.image_name)):
            self.image_num = int(max(regex.findall(self.image_name)))
        self.set_parameter()
        self.i_thread.start_image_process_thread_signal.emit(
            self.parameters,
            self.image_num, image_path_name, self.flip)

        # automatically select the csv path
        for root, dirs, files in os.walk(self.image_path):
            csv_files = []
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append()
            if not csv_files:
                QMessageBox.warning(
                    self.ui,
                    'CSV File Not Exist',
                    'There is no CSV data for the selected image. ' +
                    'Analysis might be executed without physical positions.')
            if len(csv_files) > 1:
                QMessageBox.warning(
                    self.ui,
                    'Multiple CSV Files',
                    'There are multiple CSV files. ' +
                    'Not possible to analyze with inaccurate data.')
            if len(csv_files) == 1:
                csv_path = os.path.join(self.image_path, csv_files[0])
                self.i_thread.neuron_data.position_path = csv_path
                self.i_thread.assignment.unit = 1 if self.i_thread.neuron_data.position_header == 1 else self.i_thread.neuron_data.position_header[2]

    def button_next(self):
        self.image_num += 1
        path = os.path.join(self.image_path, str(self.image_num) + ".tif")
        if not os.path.exists(path):
            path = os.path.join(self.image_path, f'{self.image_num:04}' + '.tif')
        self.set_parameter()
        self.i_thread.start_image_process_thread_signal.emit(
            self.parameters,
            self.image_num, path, self.flip)

    def button_last(self):
        self.image_num -= 1
        path = os.path.join(self.image_path, str(self.image_num) + ".tif")
        if not os.path.exists(path):
            path = os.path.join(self.image_path,
                                f'{self.image_num:04}' + '.tif')
        self.set_parameter()
        self.i_thread.start_image_process_thread_signal.emit(
            self.parameters,
            self.image_num, path, self.flip)

    def button_kill(self):
        self.i_thread.is_killed = True
        self.i_thread.neuron_data.save_data(self.save_path)

    def button_pause(self):
        if not self.i_thread.is_paused:
            self.i_thread.is_paused = True
            self.ui.button_pause.setText("Resume")
        else:
            self.i_thread.is_paused = False
            self.ui.button_pause.setText("Pause")

    def button_go(self):
        self.image_num = int(self.ui.textEdit_num.toPlainText())
        path = os.path.join(self.image_path, str(self.image_num) + ".tif")
        if not os.path.exists(path):
            path = os.path.join(self.image_path,
                                f'{self.image_num:04}' + '.tif')
        self.set_parameter()
        self.i_thread.start_image_process_thread_signal.emit(
            self.parameters,
            self.image_num, path, self.flip)

    def button_run(self):
        self.results = []
        start = int(self.ui.textEdit_start.toPlainText())
        end = int(self.ui.textEdit_end.toPlainText())
        self.image_num = start
        self.set_parameter()
        self.i_thread.is_killed = False

        self.image_nums = []
        self.right_brightness = []
        self.left_brightness = []
        self.brightness = []

        path = os.path.join(self.image_path, str(self.image_num) + ".tif")
        if not os.path.exists(path):
            path = os.path.join(self.image_path,
                                f'{self.image_num:04}' + '.tif')
        # for i in range(start, end + 1):
        # if self.stop:
        #     break
        self.i_thread.loop_signal.emit(self.parameters,
                                       self.image_num, path,
                                       self.flip, start, end)
        self.image_num += 1

    def button_refresh(self, bias_row=0, bias_column=0):
        self.set_parameter()
        self.parameters.row_bias += bias_row
        self.parameters.column_bias += bias_column
        self.initialization_parameter()

        path = os.path.join(self.image_path, str(self.image_num) + ".tif")
        if not os.path.exists(path):
            path = os.path.join(self.image_path,
                                f'{self.image_num:04}' + '.tif')
        self.i_thread.start_image_process_thread_signal.emit(
            self.parameters,
            self.image_num, path, self.flip)

    def checkbox_mirror_symmetry(self):
        self.flip = not self.flip
        self.set_parameter()

        path = os.path.join(self.image_path, str(self.image_num) + ".tif")
        if not os.path.exists(path):
            path = os.path.join(self.image_path,
                                f'{self.image_num:04}' + '.tif')
        self.i_thread.start_image_process_thread_signal.emit(
            self.parameters,
            self.image_num, path, self.flip)

    def initialization_parameter(self):
        self.ui.textEdit_alpha.setText(str(self.parameters.alpha))
        self.ui.textEdit_beta.setText(str(self.parameters.beta))

        self.ui.textEdit_peak_ratio.setText(str(self.parameters.peak_ratio))
        self.ui.textEdit_peak_circle.setText(str(self.parameters.peak_circle))

        self.ui.textEdit_right_ratio.setText(str(self.parameters.right_ratio))
        self.ui.textEdit_right_circle.setText(str(self.parameters.right_circle))

        self.ui.textEdit_row_bias.setText(str(self.parameters.row_bias))
        self.ui.textEdit_column_bias.setText(str(self.parameters.column_bias))

        self.ui.textEdit_left_ratio.setText(str(self.parameters.left_ratio))
        self.ui.textEdit_left_circle.setText(str(self.parameters.left_circle))

        self.ui.textEdit_right_black_bias \
            .setText(str(self.parameters.right_black_bias))
        self.ui.textEdit_left_black_bias \
            .setText(str(self.parameters.left_black_bias))

    def set_parameter(self) -> None:
        """This method imports parameter changes from the GUI to Parameters()"""
        self.parameters.alpha = int(self.ui.textEdit_alpha.toPlainText())
        self.parameters.beta = int(self.ui.textEdit_beta.toPlainText())

        self.parameters.peak_ratio \
            = float(self.ui.textEdit_peak_ratio.toPlainText())
        self.parameters.peak_circle \
            = int(self.ui.textEdit_peak_circle.toPlainText())

        self.parameters.right_ratio \
            = float(self.ui.textEdit_right_ratio.toPlainText())
        self.parameters.right_circle \
            = int(self.ui.textEdit_right_circle.toPlainText())

        self.parameters.row_bias \
            = int(self.ui.textEdit_row_bias.toPlainText())
        self.parameters.column_bias \
            = int(self.ui.textEdit_column_bias.toPlainText())

        self.parameters.left_ratio \
            = float(self.ui.textEdit_left_ratio.toPlainText())
        self.parameters.left_circle \
            = int(self.ui.textEdit_left_circle.toPlainText())

        self.parameters.right_black_bias \
            = int(self.ui.textEdit_right_black_bias.toPlainText())
        self.parameters.left_black_bias \
            = int(self.ui.textEdit_left_black_bias.toPlainText())

    def show_image(self, q_pixmap, img_inform: ImageInform):
        # 用于显示图片
        self.ui.label_image.setPixmap(q_pixmap)
        # 用于刷新image information
        self.img_inform = img_inform
        self.set_result()
        self.ui.box_neuron_amount.setText(str(self.i_thread.neuron_data.amount))
        self.results.append(img_inform)
        self.draw_brightness(img_inform)
        self.draw_position(img_inform)

    def show_image_loop(self, q_pixmap, img_inform: ImageInform):
        self.ui.label_image.setPixmap(q_pixmap)
        self.img_inform = img_inform
        self.set_result()
        self.results.append(img_inform)
        self.draw_brightness(img_inform)
        self.draw_position(img_inform)
        self.write_csv(img_inform, self.dataframe)

    def write_csv(self, img_inform: ImageInform, dataframe):
        # D:\UQ\DECO7861_master_thesis\Wormanalysis\analysis_gui.py:311:
        # FutureWarning: The frame.append method is deprecated and will be
        # removed from pandas in a future version. Use pandas.concat instead.
        # dataframe.append(pd.DataFrame({
        # (currently v1.4.3)
        dataframe = \
            dataframe.append(pd.DataFrame({
                'Right_row': [img_inform.right_row],
                'Right_column': [img_inform.right_column],
                'Right_brightness': [img_inform.right_brightness],
                'Left_row': [img_inform.left_row],
                'Left_column': [img_inform.left_column],
                'Left_brightness': [img_inform.left_brightness],
                'Brightness': [img_inform.brightness]}),
                ignore_index=True)

        dataframe.to_csv(self.save_path, sep=',', encoding='utf-8')

    def open_position_file(self):
        self.position_file, _ \
            = QFileDialog.getOpenFileName(self, 'Select position file', '',
                                          'Position_file(*.csv)')
        data = pd.read_csv(self.position_file, header=None)
        data = data.values
        scale = 1.1
        self.x = []
        self.y = []

        for line in data:
            self.x.append(line[0] + (line[2] - 860) * scale)
            self.y.append(line[1] + (line[3] - 600) * scale)

        self.ui.MplWidget_2.canvas.axes.clear()
        self.ui.MplWidget_2.canvas.axes.scatter(self.x, self.y)
        self.ui.MplWidget_2.canvas.axes.set_title('Position')
        self.ui.MplWidget_2.canvas.draw()

    def draw_position(self, img_inform: ImageInform):
        length = img_inform.num
        x = self.x[:length]
        y = self.y[:length]

        self.ui.MplWidget_2.canvas.axes.clear()
        self.ui.MplWidget_2.canvas.axes.scatter(x, y)

        self.ui.MplWidget_2.canvas.axes.set_title('Position')
        self.ui.MplWidget_2.canvas.draw()

    def draw_brightness(self, img_inform: ImageInform):

        self.image_nums.append(img_inform.num)
        self.right_brightness.append((img_inform.right_brightness))
        self.left_brightness.append((img_inform.left_brightness))
        # self.brightness.append((result_dict['brightness']))

        self.ui.MplWidget.canvas.axes.clear()
        self.ui.MplWidget.canvas.axes.plot(self.image_nums,
                                           self.right_brightness)
        self.ui.MplWidget.canvas.axes.plot(self.image_nums,
                                           self.left_brightness)
        # self.ui.MplWidget.canvas.axes.plot(self.image_nums, self.brightness)
        self.ui.MplWidget.canvas.axes.legend(('Right', 'Left'),
                                             loc='upper right')
        self.ui.MplWidget.canvas.axes.set_title('Brightness')
        self.ui.MplWidget.canvas.draw()

    def set_result(self):
        self.initialization_parameter()
        self.ui.textEdit_num.setText(str(self.img_inform.num))

        self.ui.text_right_coordinate.setText(
            str(self.img_inform.right_row) + ':' +
            str(self.img_inform.right_column)
        )
        self.ui.text_right_brightness.setText(
            str(self.img_inform.right_brightness))

        self.ui.text_left_coordinate.setText(
            str(self.img_inform.left_row) + ':' +
            str(self.img_inform.left_column)
        )
        self.ui.text_left_brightness.setText(
            str(self.img_inform.left_brightness))

        self.ui.text_brightness.setText(str(self.img_inform.brightness))
        self.ui.text_right_black.setText(str(self.img_inform.right_black))
        self.ui.text_left_black.setText(str(self.img_inform.left_black))

    def set_neuron_amount(self):
        string = self.ui.box_neuron_amount.toPlainText()
        if string.isdigit():
            self.i_thread.neuron_data.amount = int(string)
            self.i_thread.assignment.amount = int(string)

    def swap(self):
        tag1 = self.ui.textEdit_neuron_tag_1.toPlainText()
        tag2 = self.ui.textEdit_neuron_tag_2.toPlainText()
        image_num = self.ui.textEdit_num.toPlainText()
        if tag1.isdigit() and tag2.isdigit() and image_num.isdigit():
            self.i_thread.swap_neuron_position(int(image_num), str(tag1), str(tag2))
        else:
            QMessageBox.warning(
                self.ui,
                'Swap Neuron\' Position',
                'Some of these are not integer: \n' +
                'Neuron Tag 1: ' + str(tag1.isdigit()) + ';\n'
                'Neuron Tag 2: ' + str(tag2.isdigit()) + ';\n'
                'Image Number: ' + str(tag1.isdigit()) + '.')