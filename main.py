import DataViewer
import os
import numpy as np
def main():
    ImagePath = "../busesTrain"
    AnnotationFile = "../annotationsTrain.txt"
    pDV = DataViewer.CDataViewer()
    ImageList = pDV.LoadImages(ImagePath)
    DetectedObjects = pDV.LoadAnnotations(AnnotationFile)
    for ImageFullPathFile in ImageList:
        FileName = os.path.basename(ImageFullPathFile)
        BoxList = DetectedObjects[FileName]
        if BoxList:
            pDV.PlotBoxes(ImageFullPathFile,BoxList)


        pass

    pass

if __name__ == "__main__":
    main()