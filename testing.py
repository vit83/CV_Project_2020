
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import torchvision.transforms as T
from engine import train_one_epoch, evaluate
import cv2



def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(torch.cuda.get_device_capability(0))
    num_classes = 2
    model = get_model_instance(num_classes)

    # move model to the right device

    model.load_state_dict(torch.load("busModel.pth"))
    model.to(device)
    model.eval()
    object_detection_api('test.jpg', threshold=0.8,model = model)
    pass

def get_model_instance(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_prediction(img_path, threshold,model):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to('cuda')
    torch.cuda.synchronize()

    pred = model([img])
    #convert from tensor gpu to tensor cpu in order to use numpy and access data
    pred = [{k: v.to("cpu") for k, v in t.items()} for t in pred]
    #pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = []
    detectedBoxes = []
    for x in pred_score:
        if x > threshold:
            pred_t.append(pred_score.index(x))
            detectedBoxes.append(pred_boxes[pred_score.index(x)])
    if (detectedBoxes == []):
        return None


    return (detectedBoxes)

def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3,model = None):
    """
    object_detection_api
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
      method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written
          with opencv
        - the final image is displayed
    """
    boxes = get_prediction(img_path, threshold,model)
    if (boxes is None):
            print("No detection")
            return
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()











if __name__ == "__main__":
    main()
