from PIL import Image
import numpy as np

import os
import OsUtils
import DataViewer
import math
import imgaug.augmenters as iaa
import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from matplotlib import pyplot as plt
def main():

    ImagePath = "./busesTrain"
    AnnotationFile = "./annotationsTrain.txt"
    AnnotationOutputFile = "./annotationsTrainAug.txt"
    OutImgPath = "./BusesAug"
    pDV = DataViewer.CDataViewer()

    OsUtils.remove(OutImgPath)
    os.mkdir(OutImgPath)
    ImageList = pDV.LoadImages(ImagePath)
    LabelsDict = pDV.LoadAnnotations(AnnotationFile)
    LabelsDictAug = GetBoundingBoxAndLabel(LabelsDict)


    #ia.seed(1)
    NumberOfAugRuns = 100
    seq = iaa.Sequential([
        #iaa.Affine(rotate=(-90, 90)),
        #iaa.AdditiveGaussianNoise(scale=(10, 60)),
        #iaa.GaussianBlur(sigma=(0, 0.5)),
        #iaa.ContrastNormalization((0.75, 1.5)),
        #iaa.Multiply((0.8, 1.2), per_channel=0.2),
        #iaa.Crop(percent=(0, 0.2))
        #iaa.fliplr(0.5),
        iaa.Affine(
            scale=(0.5, 1.5),
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-180, 180),
            )
    ], random_order=True)
    AugRes = {}
    for i in range(1,NumberOfAugRuns+1):
        print("iteration  {:d}".format(i))
        for img_path in ImageList:
            fileName = os.path.basename(img_path)
            Boxes , Labels = LabelsDictAug[fileName]
            img = imageio.imread(img_path)
            bbs = BoundingBoxesOnImage(Boxes, shape=img.shape)
            fileNameAug = "Aug" + str(i) + "_" + fileName


            images_aug ,bbs_aug  =  seq(image=img, bounding_boxes=bbs)

            #image_before = bbs.draw_on_image(img, size=4)
            #image_after = bbs_aug.draw_on_image(images_aug, size=4, color=[0, 0, 255])
            #ia.imshow(image_before)
            #ia.imshow(image_after)

            np_image = np.hstack([images_aug])
            #ia.imshow(np_image)
            im = Image.fromarray(np_image)
            saveStr = OutImgPath +"\\" + fileNameAug
            im.save(saveStr)
            AugRes[fileNameAug] = BoxTolist(bbs_aug)

    WriteResults(AugRes,AnnotationOutputFile)

def BoxTolist(bbs):
    BoxList = []
    for i in range(len(bbs.bounding_boxes)):
        box = bbs.bounding_boxes[i]
        BoxObj = [box.x1,box.y1 , box.x2 , box.y2,box.label]
        BoxList.append(BoxObj)
    return  BoxList


def GetBoundingBoxAndLabel(LabelsDict):
    LabelsDictAug = {}
    for key, boxlist in LabelsDict.items():
        boxes = []
        labels = []
        for value in boxlist:
            # Create a Rectangle patch
            x = value[0]
            y = value[1]
            w = value[2]
            l = value[3]
            label = value[4]
            x1 = x
            y1 = y
            x2 = x1 + w
            y2 = y1 + l
            CurrentBox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2,label = label)
            boxes.append(CurrentBox)
            labels.append(label)
        LabelsDictAug[key] = [boxes, labels]
    return LabelsDictAug

def WriteResults(Results,file):
    with open(file, 'w+') as result_file:
        for  imageFullPath, Boxlist  in Results.items():
            FileName = os.path.basename(imageFullPath)
            line = FileName + ":"
            for box in Boxlist:
                x = math.ceil(box[0])
                y = math.ceil(box[1])
                w = math.ceil(box[2]) - x
                l =  math.ceil(box[3]) - y
                label = math.ceil(box[4])

                boxString = "[{:d},{:d},{:d},{:d},{:d}],".format(x,y,w,l,1)
                line += boxString
            #remove last comma
            line = line[:-1] + '\n'
            result_file.writelines(line)

if __name__ == "__main__":
    main()