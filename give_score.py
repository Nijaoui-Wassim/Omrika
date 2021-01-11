import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32, 5)
        self.conv2 = nn.Conv2d(32,64, 5)
        self.conv3 = nn.Conv2d(64,128, 5)
       
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        #rint(x[0].shape)
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


PATH  =  'firsttModel'.replace('/','\ ').replace(' ','')
net = torch.load(PATH)
net.eval()


def normalize_score(score):
    score = 1/score
    while(score < 10):
        score *= 10
        if score< 10:
            score = 9**score
    
    while(score > 10):
        score /= 1.5
    return score


def check_pic(name):
    print(name)
    Pic_Path  =  'images/'+(name+'.jpg').replace('/','\ ').replace(' ','')
    IMG_SIZE = 50
    img = cv2.imread(Pic_Path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    
    Xx= torch.Tensor([i for i in tqdm(img_array)]).view(-1,1,50,50)#.to(device)
    Xx = Xx/255.0
    
    net_out = net(Xx[0].view(-1,1,50,50))[0]
    predicted_class = torch.argmax(net_out)
    print(predicted_class)
    #print(predicted_class.item())
    if int(predicted_class.item())== 1 :
        print('you are attractive')
    else:
        print('you are NOT')
    print(net_out)
    sc = float(net_out[1].item())
    sc = normalize_score(sc)
    print('attractive score out of 10 : ', sc)
    #print(' % : ', print(net_out[1].item()))
    print('\n\n ' + '='*40 + '\n\n')
    return sc



def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


    
def run():
    ranking = {}

    
    TestPath =  'images/'.replace('/','\ ').replace(' ','')
    for file in get_files(TestPath):
        if file.endswith(".jpg"):
            try:
                ranking[file] = check_pic(file.split('.')[0])
            except Exception as e:
                print(e)

    #print (ranking)

    max_score = max(ranking, key=ranking.get) 

    print (max_score)
    print (ranking[max_score])
    return ranking
    #show(max_score)

    
def show(picture):

    Pic_Path = 'images/'.replace('/','\ ').replace(' ','')
    Pic_Path += picture #+ '.jpg'

    from PIL import Image
    image = Image.open(Pic_Path)
    image.show()


if __name__ == "__main__":
    run()
