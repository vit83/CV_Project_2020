import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import shutil
import os
import OsUtils
class CDataViewer:
    def __init__(self):
        self.ImageFiles = []
        self.objects = {}

        pass
    def LoadImages(self,ImageDirPath,filetpye = '.JPG'): #to do add support for list of file types and uppercase/lower case invariance
        self.ImageFiles = OsUtils.GetFilesFromDir(ImageDirPath,filetpye)
        return  self.ImageFiles
    def LoadAnnotations(self,AnnoFilePath):
        self.AnnotationFile = AnnoFilePath
        with open(AnnoFilePath) as f:
            lines = f.read().splitlines()
        for line in lines:
            fileName , Boxes = line.split(':')
            BoxListStr =  [Boxes.split(']')[0] for Boxes in Boxes.split('[') if ']' in Boxes]
            BoxList = []
            for CurrentBox in BoxListStr:
                Box = (CurrentBox.split(','))
                Box = [int(i)for i in Box]
                BoxList.append(Box)
                pass
            self.objects[fileName] = BoxList
            pass
        return self.objects

    def PlotBoxes(self,file,BoxList):
        im = np.array(Image.open(file), dtype=np.uint8)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)
        for box in BoxList:
            # Create a Rectangle patch
            x = box[0]
            y = box[1]
            w = box[2]
            l = box[3]
            rect = patches.Rectangle((x, y), w, l, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()

    def SaveBoxesAsImages(self,file,BoxList,ImageOutPutDir,bDisplay = False):
        i = 1
        for box in BoxList:
            # Create a Rectangle patch
            x = box[0]
            y = box[1]
            w = box[2]
            l = box[3]
            label = box[4]
            img = cv2.imread(file)
            crop_img = img[y:y + l, x:x + w]
            if crop_img.size == 0:
                continue
            if bDisplay:
                cv2.imshow("cropped", crop_img)
                cv2.waitKey(0)
            FileName = os.path.basename(file)
            Name , Type = FileName.split('.')
            Name = str(label)+'_' + Name + '_' + str(label) + '.' + Type
            Name = ImageOutPutDir + '\\' + str(label) +"\\"+ Name
            cv2.imwrite(Name, crop_img)
            i = i + 1
        pass


