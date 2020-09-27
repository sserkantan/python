# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:37:59 2020

@author: SSTAN


import sys
from PyQt5 import QtWidgets
app=QtWidgets.QApplication(sys.argv)
pencere=QtWidgets.QWidget()
pencere.setWindowTitle("Window Name")
pencere.setFixedSize(300,300)
pencere.setStyleSheet("background-color :red")

pencere.show()
sys.exit(app.exec_())
"""

import sys
from PyQt5 import QtWidgets
class Pencere(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.gui()
    def gui(self):
        self.etiket = QtWidgets.QLabel(self)
        self.etiket.setText("merhaba")
        self.show()
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    pencere = Pencere()
    sys.exit(app.exec_())