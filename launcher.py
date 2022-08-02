import PySide2
from PySide2.QtWidgets import *

import os
from analysis_gui import MainWidget

if __name__ == '__main__':
    dirname = os.path.dirname(PySide2.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    print("plugin path:", plugin_path)

    app = QApplication([])
    window = MainWidget()
    window.show()
    app.exec_()