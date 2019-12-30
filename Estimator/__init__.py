import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from keras.models import load_model
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
DetectionModel = get_model_instance(num_classes)
print("loading detection model")
DetectionModel.load_state_dict(torch.load("busModel.pth"))
DetectionModel.to(device)
DetectionModel.eval()
print("done loading model")
print("warming up  model")
img = Image.open('warmup.JPG')
transform = T.Compose([T.ToTensor()])
img = transform(img).to('cuda')
torch.cuda.synchronize()
pred = DetectionModel([img])
print("model is ready")
print("loading color classification model")
ColorModel = load_model('color_model.h5')
print("model is ready")
def getDetectionModel():
    return DetectionModel

def getColorClassificationModel():
    return ColorModel