import warnings
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataloader import *
from models.model import *
from utils import *
from torchvision import transforms
import torch
import torch.nn.functional as F
import os
testroot = "/root/pycharm/ResNet_RockRecognition_keras/data/underglass_rock_recognization/val/"
# 创建模型
model = torchvision.models.densenet121()
# 全连接层
model.fc = nn.Linear(1000, 30)
model.load_state_dict(torch.load("checkpoints/best_model/densenet121/0/model_best.pth")["state_dict"])
model.cuda()
# model = torch.load("checkpoints/best_model/densenet121/0/model_best.pth")
# model.cuda()
# print(model)
torch.no_grad()
print(model)
print('model load finish')

def predict():
    pred = []
    target = []
    dirlist = []
    dirs = os.listdir(testroot)
    for dir in dirs:
        dirpath = os.path.join(testroot, dir)
        imgfiles = os.listdir(dirpath)
        dirlist.append(dir)
        # print(dir)
        dirpred = []
        for img in imgfiles:
            imgpath = os.path.join(dirpath, img)
            img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (320, 320))  # 是否需要resize取决于新图片格式与训练时的是否一致
            img = np.transpose(img, (2, 0, 1))
            input = torch.from_numpy(img)
            input = input.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float32)
            input = input.unsqueeze(0)
            # print(img.shape)
            outputs = model(input)# outputs，out1修改为你的网络的输出
            outputs = F.softmax(model.fc(outputs))
            print(outputs)
    #         predicted = torch.argmax(outputs, 1).cpu().numpy()
    #         pred.append(predicted[0])
    #         dirpred.append(predicted[0])
    #     label = np.argmax(np.bincount(np.array(dirpred)))
    #     for i in range(len(dirpred)):
    #         target.append(label)
    # return pred, target, dirlist
            # print(predicted)
            # predicted, index = torch.max(out1, 1)
            # degre = int(index[0])
            # list = [0, 45, -45, -90, 90, 135, -135, 180]
            # print(predicted, list[degre])
preds, targets, classmembers= predict()
print(preds)
print(targets)
print(classmembers)
# predict()