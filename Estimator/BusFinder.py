from PIL import Image
from . import getModel
import torch
import torchvision.transforms as T

def get_prediction(img_path):

    threshold = 0.8
    model = getModel()
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to('cuda')
    torch.cuda.synchronize()
    pred = model([img])
    #convert from tensor gpu to tensor cpu in order to use numpy and access data
    pred = [{k: v.to("cpu") for k, v in t.items()} for t in pred]
    #pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())]
    # convert from x1,y1,x2,y2 to x1,y1,w,l
    pred_boxes = [[i[0], i[1], i[2] - i[0] , i[3] - i[1]] for i in list(pred[0]['boxes'].detach().numpy())]
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


