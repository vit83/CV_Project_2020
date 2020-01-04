import cv2
import numpy as np
import OsUtils
import os

import imgaug as ia
import imgaug.augmenters as iaa
class CDataAugmentation:
    def __init__(self):
        pass
    def flip(self,ImageDir,OutputImageDir):
        files = OsUtils.GetFilesFromDir(ImageDir,'.JPG')
        for file in files:
            originalImage = cv2.imread(file)

            flipVertical = cv2.flip(originalImage, 0)
            flipHorizontal = cv2.flip(originalImage, 1)
            flipBoth = cv2.flip(originalImage, -1)

            FileName = os.path.basename(file)
            Name , Type = FileName.split('.')

            VericalName = Name + '_V' +  '.' + Type
            HoriznotalName = Name + '_H' +  '.' + Type
            BothName = Name + '_VH' +  '.' + Type

            SavePath = OutputImageDir + '\\' + VericalName
            cv2.imwrite(SavePath, flipVertical)

            SavePath = OutputImageDir + '\\' + HoriznotalName
            cv2.imwrite(SavePath, flipHorizontal)

            SavePath = OutputImageDir + '\\' + BothName
            cv2.imwrite(SavePath, flipBoth)

        pass

    def rotate(self,ImageDir,OutputImageDir):
        files = OsUtils.GetFilesFromDir(ImageDir, '.JPG')
        for file in files:
            originalImage = cv2.imread(file)

            Rotate90  = cv2.rotate(originalImage,cv2.ROTATE_90_CLOCKWISE)
            Rotate270 = cv2.rotate(originalImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

            FileName = os.path.basename(file)
            Name, Type = FileName.split('.')

            R90Name = Name + '_R90' + '.' + Type
            R270Name = Name + '_R270' + '.' + Type


            SavePath = OutputImageDir + '\\' + R90Name
            cv2.imwrite(SavePath, Rotate90)

            SavePath = OutputImageDir + '\\' + R270Name
            cv2.imwrite(SavePath, Rotate270)




