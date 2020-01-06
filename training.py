# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
import DataViewer


def main():


    ImagePath = "./busesTrain"
    AnnotationFile = "./annotationsTrainOneClass.txt"
    ImagePath = "./BusesAug"
    AnnotationFile = "./annotationsTrainAug.txt"
    pDV = DataViewer.CDataViewer()
    ImageList = pDV.LoadImages(ImagePath)
    Labels = pDV.LoadAnnotations(AnnotationFile)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(torch.cuda.get_device_capability(0))
    #device = torch.device('cpu')


    # our dataset has two classes only - background and person
    #6 buses and one for all others
    #num_classes = 6 + 1
    num_classes = 2
    num_epochs = 4
    # use our dataset and defined transformations
    dataset = BussesDataset(ImageList,Labels, get_transform(train=True))
    dataset_test = BussesDataset(ImageList,Labels, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    NumberOfItem = len(indices)
    NumberOfItemForTrain = int(0.9*NumberOfItem)
    NumberOfItemForTest = NumberOfItem - NumberOfItemForTrain
    dataset = torch.utils.data.Subset(dataset, indices[:NumberOfItemForTrain])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[NumberOfItemForTrain:])

    # define training and validation data loaders
    #TO-DO : need to implement my own data loader

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=utils.collate_fn,pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        collate_fn=utils.collate_fn,pin_memory=True)

    # get the model using our helper function
    model = get_model_instance(num_classes)
    print(model)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    # let's train it for 10 epochs


    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        saveStr = "busModelV9" + str(epoch) + ".ptn"
        torch.save(model.state_dict(), saveStr)
    torch.save(model.state_dict(), "busModelV9.pth")
    print("That's it!")

class BussesDataset(torch.utils.data.Dataset):
    def __init__(self, images,labels, transforms):
        self.transforms = transforms
        self.imgs = images
        self.annotations = labels


    def __getitem__(self, idx):
        # load images ad masks
        img_path =  self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        FileName = os.path.basename(img_path)
        BoxList = self.annotations[FileName]
        # get bounding box coordinates for each mask based on mask analysis

        boxes = []
        labels = []
        for box in BoxList:
            # Create a Rectangle patch
            x = box[0]
            y = box[1]
            w = box[2]
            l = box[3]
            label = box[4]
            boxes.append([x, y, x + w, y + l])
            labels.append(label)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        NumberOfBoxes = len(BoxList)
        iscrowd = torch.zeros((NumberOfBoxes), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)




def get_model_instance(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
    #    transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



    
if __name__ == "__main__":
    main()
