# -*- coding: utf-8 -*-
"""
Project : Zebrafish
Created on Fri May 15 16:48:33 2020
Description: GUI to select a folder of control images , 
                a folder of subject images
                and view the iou statistics comparison for control & subject
"""

import zebrafish_group_comparison
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QMovie
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)

class GWidget(QMainWindow):
    
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        super(GWidget, self).__init__()
        
        # Directory locations of control and subject folders from browsing
        self.control_dir = ""
        self.subject_dir = ""   
        
        # Labels for description of images in GUI
        self.tLabel1 = QLabel()
        self.tLabel2 = QLabel()
        self.tLabel3 = QLabel()
        
        # Including scrolling on widget to view all the images in the directory
        self.wid = QWidget()
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.wid)
        self.setCentralWidget(self.scroll)
        
        # Label to plot the output comparison plot
        self.iLabel3 = QLabel()       
        
        # Label to set title
        self.title = QLabel()
        
        # Grid Layout
        self.gbox = QGridLayout(self.wid)
        
        # Function for rendering GUI
        self.setThings()
        
        
    def setThings(self):
        
        # Setting a title
        self.title.setText("Zebrafish Project")
        self.title.setStyleSheet("text-align:center; background-color: black; color:blue; font-size:50px; padding: 0 0 0 0px; margin: 0 0 0 0px;")

        # Get control button properties and calling on_click function
        get_control = QPushButton('Select Control Images Directory', self)
        get_control.clicked.connect(lambda: self.on_click("control"))
        self.gbox.addWidget(get_control, 1,0)
        
        # Get subject button properties and calling on_click function
        get_subject = QPushButton('Select Subject Images Directory', self)
        get_subject.clicked.connect(lambda: self.on_click("subject"))
        self.gbox.addWidget(get_subject, 1,1)
        
        # Generate button properties and calling on_click function
        generate_button = QPushButton('Generate', self)
        generate_button.clicked.connect(lambda: self.generate(self.control_dir, self.subject_dir))
        self.gbox.addWidget(generate_button, 1,2)
        
        # setting positions of different labels on grid layout
        self.gbox.addWidget(self.iLabel3, 3,2)
        self.gbox.addWidget(self.tLabel1, 2,0)
        self.gbox.addWidget(self.tLabel2, 2,1)
        self.gbox.addWidget(self.tLabel3, 2,2)

        
        self.setGeometry(10, 100, QDesktopWidget().screenGeometry().width(), QDesktopWidget().screenGeometry().height())
    
    # On click function for get_control and get_subject
    # argument f values - control/subject
    @pyqtSlot()
    def on_click(self, f):
        i = 0
        if(f=="control"):
            i=0
            dirName = QFileDialog.getExistingDirectory(self, "select dir")
            self.control_dir=dirName
            self.tLabel1.setText(dirName)
        else:
            i=1
            dirName = QFileDialog.getExistingDirectory(self, "select dir")
            self.subject_dir=dirName
            self.tLabel2.setText(dirName)
        
        # A new label gets created for every image in the directory
        dir_list = os.listdir(dirName)
        for index, file in enumerate(dir_list):
            print(file)
            tempLabel = QLabel()
            image = QImage(os.path.join(dirName,file))
            tempLabel.setPixmap(QPixmap.fromImage(image.scaled(800,800)))
            self.gbox.addWidget(tempLabel, index+3, i)
    
    # on click function for generate_button to generate comparison plot
    #using images from directories selected (f1, f2)
    def generate(self, f1, f2):
        self.tLabel3.setText("Plot of Registration Stats")
        self.tLabel3.setStyleSheet("font-size:30px;qproperty-alignment: AlignCenter;")
        
        # Call to align each of these images with control and subject and calculate iou to plot
        zebrafish_group_comparison.main(f1, f2)
        fileName = "iou_plot.png"
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
        self.iLabel3.setPixmap(QPixmap.fromImage(image.scaled(1200,1000)))

def main():
    import sys
    
if __name__ == "__main__":
    main()
