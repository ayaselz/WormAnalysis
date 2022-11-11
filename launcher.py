"""
The launcher for Worm Analyser. To run this software, click the green right
arrow with an alternate text "Run 'launcher'", or right click this interface
and select "Run 'launcher'", or click the green right arrow on the code line 24
and select "Run 'launcher'".

蠕虫分析器的启动器。 要运行此软件，请单击带有备用文本“Run 'launcher'”的绿色右箭头，
或右键单击此界面并选择“Run 'launcher'"，或单击代码行24上的绿色右箭头并选择“Run 'launcher'"。
"""

import PySide2
from PySide2.QtWidgets import *

import os
from analysis_gui import MainWidget


__author__ = "{{Yuliang Liu}} and {{Wencan Peng}} ({{46222378}})"
__email__ = "wencan.peng@uqconnect.edu.au"
__date__ = "16/08/2022"
__version__ = "2.0.0"
__doc__ = "link"

if __name__ == '__main__':
    dirname = os.path.dirname(PySide2.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    # print("plugin path:", plugin_path)

    app = QApplication([])
    window = MainWidget()
    window.show()
    app.exec_()

