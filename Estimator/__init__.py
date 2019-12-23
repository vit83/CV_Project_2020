import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#startup code to save estimation time later
def get_model_instance(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.get_device_capability(0))
num_classes = 2
model = get_model_instance(num_classes)
print("loading model")
model.load_state_dict(torch.load("busModel.pth"))
model.to(device)
model.eval()
print("done loading model")

def getModel():
    return model