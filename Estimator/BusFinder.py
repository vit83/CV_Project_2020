from PIL import Image
from . import getDetectionModel , getColorClassificationModel
import torch
import torchvision.transforms as T
import numpy as np
from matplotlib import pyplot as plt

def get_prediction(img_path):

    threshold = 0.8
    DetectionModel = getDetectionModel()
    img = Image.open(img_path)
    print(img.size)
    #plt.imshow(img)
    #plt.show()

    #img2 = img.copy() #40 ms
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to('cuda')
    print(img.shape)
    torch.cuda.synchronize()
    pred = DetectionModel([img])
    #convert from tensor gpu to tensor cpu in order to use numpy and access data
    pred = [{k: v.to("cpu") for k, v in t.items()} for t in pred]
    #pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())]
    # convert from x1,y1,x2,y2 to x1,y1,w,l
    pred_boxes = [[i[0], i[1], i[2] - i[0] , i[3] - i[1]] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_labels = list(pred[0]["labels"].detach().numpy())
    pred_t = []
    detectedBoxes = []
    #detectedLabels = []
    for x in pred_score:
        if x > threshold:
            pred_t.append(pred_score.index(x))
            detectedBoxes.append(pred_boxes[pred_score.index(x)])
            #detectedLabels.append(pred_labels[pred_score.index(x)])
    if (detectedBoxes == []):
        return None
    NumberOfBoxes,_ = np.shape(detectedBoxes)
    detectedBoxesWithColor = np.zeros([NumberOfBoxes,5])
    i = 0
    for Box in detectedBoxes:
        Box = (np.ceil(Box)).astype(int)
        DetectedObject = img[:,Box[1]:(Box[1]+Box[3]),Box[0]:(Box[0]+Box[2])]
        DetectedImg = (T.ToPILImage()(DetectedObject.cpu())).resize((224,224))
        color = get_color(DetectedImg)
        Box = np.append(Box,color)
        detectedBoxesWithColor[i] = Box
        i = i + 1
        #plt.imshow(DetectedImg)
        #plt.show()
        #plt.imshow([[(res)]])
        #plt.show()

    return (detectedBoxesWithColor)


def get_color(DetectedImg):
    ColorModel = getColorClassificationModel()
    img_tensor = np.expand_dims(DetectedImg, axis=0)
    img_tensor2 = img_tensor / 255.0
    pred = ColorModel.predict(img_tensor2)
    color = np.argmax(pred) + 1
    return color


