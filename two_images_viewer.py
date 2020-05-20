# -*- coding: utf-8 -*-
"""
Project : Zebrafish
Author: Vijay Sai Kondamadugu
Created on Fri May 15 16:48:33 2020
Description: GUI to select a template image , a subject image
                and see their ovelap after registering subject to template
    
"""
import zebrafish_alignment_oneimage
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)

class TWidget(QWidget):
    
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        super(TWidget, self).__init__()

        # Labels for displaying images 
        self.iLabel1 = QLabel(self)
        self.iLabel2 = QLabel(self)
        self.iLabel3 = QLabel(self)
        
        # File names of template and subject image retrived from browsing folders
        self.template_file = ""
        self.subject_file = ""
        
        # Labels for description of images in GUI
        self.tLabel1=QLabel(self)
        self.tLabel2=QLabel(self)
        self.tLabel3=QLabel(self) 
        
        # Function for rendering GUI
        self.setThings()
        
        
    def setThings(self):
        
        # Grid Layout
        gbox = QGridLayout(self)

        # Get template button properties and on_click function
        get_template_button = QPushButton('Get Template Image', self)
        get_template_button.clicked.connect(lambda: self.on_click("template"))
        gbox.addWidget(get_template_button, 0, 0)
        
        # Get template button properties and on_click function
        get_subject_button = QPushButton('Get Subject Image', self)
        get_subject_button.clicked.connect(lambda: self.on_click("subject"))
        gbox.addWidget(get_subject_button, 0, 1)
        
        # Registeration button properties and on_click function
        register_button = QPushButton('Register', self)
        register_button.clicked.connect(lambda: self.generate(self.template_file, self.subject_file))
        gbox.addWidget(register_button, 0, 2)

        gbox.addWidget(self.tLabel1, 1,0)
        gbox.addWidget(self.tLabel2, 1,1)
        gbox.addWidget(self.tLabel3, 1,2)

        gbox.addWidget(self.iLabel1, 2,0)
        gbox.addWidget(self.iLabel2, 2,1)
        gbox.addWidget(self.iLabel3, 2,2)
        
        # Configuring grid layout
        gbox.setRowStretch(0,1)
        gbox.setRowStretch(1,1)
        gbox.setRowStretch(2,5)
        gbox.setRowStretch(3,10)

        d = QDesktopWidget().screenGeometry()
        self.setGeometry(10, 100, d.width(), d.height())


    # On click function for get_templete and get_subject
    # argument f values - template/subject
    @pyqtSlot()
    def on_click(self, f):
        if(f == "template"):
            label = self.iLabel1
        else:
            label = self.iLabel2
            
        # Registeration button properties and on_click function
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        print("f=",fileName)
        # Generating image data compatible with PyQt5 for display
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return

            # Display of images and description of images on GUI
            if(f=="template"):
                self.template_file=fileName
                self.tLabel1.setText("Template image")
                self.tLabel1.setStyleSheet("font-size:50px; qproperty-alignment: AlignCenter;")
            else:
                self.subject_file=fileName
                self.tLabel2.setText("subject image")
                self.tLabel2.setStyleSheet("font-size:50px;qproperty-alignment: AlignCenter;")
            label.setPixmap(QPixmap.fromImage(image))
            label.setStyleSheet("qproperty-alignment: AlignCenter;")
            

    # on click function for register_button to generate results using images files selected (f1, f2)
    def generate(self, f1, f2):
        self.iLabel3.setText("Generating...")
        print("Started registration")
        image = QImage("loading.gif")
        self.iLabel3.setPixmap(QPixmap.fromImage(image))
        
        # Call to align template with subject
        zebrafish_alignment_oneimage.main(f1,f2)
        # Result image stored as result.png 
        fileName = "result.png"
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
        # Display of images and description of images on GUI
        self.tLabel3.setText("Subject registered to template")  
        self.tLabel3.setStyleSheet("font-size:50px;qproperty-alignment: AlignCenter;")
        self.iLabel3.setPixmap(QPixmap.fromImage(image))
        self.iLabel3.setStyleSheet("qproperty-alignment: AlignCenter;")

def main():
    import sys
 

if __name__ == "__main__":
    main()
 