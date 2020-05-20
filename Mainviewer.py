# -*- coding: utf-8 -*-
"""
Project : Zebrafish
Created on Sun May 17 16:52:52 2020
Description: GUI - Main landing page of application which allows user to select
                either two_images_viewer or group_comparison_viewer
"""

import two_images_viewer
import group_comparison_viewer
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)

class MWidget(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
    
        super(MWidget, self).__init__()
        
        # Grid Layout
        self.gbox = QGridLayout(self)
        
        # Buttons to run each of the applications
        self.two_image_button = QPushButton("Run Two Images", self)
        self.group_image_button = QPushButton("Run group comparison", self)
        
        # Label to set the title bar of the main application
        self.title = QLabel(self)
        self.title0 = QLabel(self)
        self.title2 = QLabel(self)
        
        self.ui_func()
        
    def ui_func(self):
        
        # Set title
        self.title.setText("Zebrafish Project")
        self.title.setStyleSheet("font-size:100px;qproperty-alignment:AlignCenter;background-color:black;color:lightblue;")
        
        # set both the buttons
        self.two_image_button.clicked.connect(lambda: self.on_click("two_image"))
        self.group_image_button.clicked.connect(lambda: self.on_click("group"))
        
        # title and button positions and styling
        self.gbox.addWidget(self.title,0,1)
        self.gbox.addWidget(self.title0,0,0)
        self.title0.setStyleSheet("background-color:black;")
        self.gbox.addWidget(self.title2,0,2)
        self.title2.setStyleSheet("background-color:black;")
        self.gbox.addWidget(self.two_image_button,2,1)
        self.gbox.addWidget(self.group_image_button,3,1)
        
        self.gbox.setRowStretch(0,2)
        self.gbox.setRowStretch(1,1)
        self.gbox.setRowStretch(2,5)
        self.gbox.setRowStretch(3,5)
        self.gbox.setRowStretch(4,10)
        
        self.setGeometry(10, 100, QDesktopWidget().screenGeometry().width(), QDesktopWidget().screenGeometry().height())
    
    # On click function for two_image_button and group_image_button
    # argument f values - two_image/group    
    @pyqtSlot()
    def on_click(self, f):
        if(f=="two_image"):
            # show two_images_viewer
            tw = two_images_viewer.TWidget(self)
            tw.show()
            
        elif(f=="group"):
            # show group_comparison_viewer
            gw = group_comparison_viewer.GWidget(self)
            gw.show()

def main():
    import sys
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    w = MWidget()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()