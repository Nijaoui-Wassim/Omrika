# Face Recognition

import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.


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

current_score = 0

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
    Pic_Path  =  (name+'.jpg').replace('/','\ ').replace(' ','')
    IMG_SIZE = 50
    img = cv2.imread(Pic_Path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    
    Xx= torch.Tensor([i for i in (img_array)]).view(-1,1,50,50)#.to(device)
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
    sc = max(float(net_out[0].item()),float(net_out[1].item()))
    #sc = normalize_score(sc)
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

to_delete=[]

def remove_files(files):
    for file in files:
        os.remove(file)


def detect(img):
    global current_score
    counter=0
    current_score = 0
    faces = face_cascade.detectMultiScale(img, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp_'+str(counter)+'.jpg', crop_img)
        current_score += check_pic('temp_'+str(counter))
        to_delete.append('temp_'+str(counter)+'.jpg')
        counter+=1
    if len(faces)>0:
        return current_score /len(faces)
    else:
        return 0



img = cv2.imread("images//fam.jpg")
count = detect(img)
print("done for ", count)
remove_files(to_delete)



#ranks = run()
