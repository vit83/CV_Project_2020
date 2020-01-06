import DataViewer
import DataAugmentation
import os
import numpy as np
import OsUtils
def main():
    ImagePath = "./busesAug"
    AnnotationFile = "./annotationsTestAug.txt"
    OutImgPath = "./BusesOnly"
    pDV = DataViewer.CDataViewer()
    pDA = DataAugmentation.CDataAugmentation()
    OsUtils.remove(OutImgPath)
    os.mkdir(OutImgPath)
    for label in range (1,7):
        os.mkdir(OutImgPath + "\\"+str(label))
    ImageList = pDV.LoadImages(ImagePath)
    DetectedObjects = pDV.LoadAnnotations(AnnotationFile)
    print("extracting data")
    for ImageFullPathFile in ImageList:
        FileName = os.path.basename(ImageFullPathFile)
        BoxList = DetectedObjects[FileName]
        if BoxList:
            #pDV.PlotBoxes(ImageFullPathFile,BoxList)
            pDV.SaveBoxesAsImages(ImageFullPathFile,BoxList,OutImgPath)
        pass
    pass
    print("finished extracting data")
    print("begining data augmentation")
    #pDA.rotate(OutImgPath, OutImgPath)
    #pDA.flip(OutImgPath,OutImgPath)
    print("finished data augmentation")
if __name__ == "__main__":
    main()
    print("done")