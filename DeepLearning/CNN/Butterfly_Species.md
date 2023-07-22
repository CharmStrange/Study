사용한 [Butterfly Species 데이터셋](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) 

**데이터셋 및 파일 접근을 위한 os, ImageFolder, Image 등의 모듈과, 직접적 데이터 조작을 위한 모듈, 연산을 위한 알고리즘을 가진 모듈을 import.
메인은 PyTorch 프레임워크 사용.**


```python
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
```

**torchvision의 transforms 메소드로 사용할 이미지 데이터에 대한 전처리 파이프라인 틀을 잡아줌.
: transforms.Compose() 메소드는 모든 이미지 데이터가 같은 전처리 과정을 거치게 함.**


```python
transform=transforms.Compose([
    transforms.RandomRotation(10), # 랜덤하게 이미지 회전(10도) : 데이터 다양하게 변형하여 과적합 방지
    transforms.RandomHorizontalFlip(), # 랜덤하게 이미지 좌우 뒤집기 : 위와 동일
    transforms.Resize(224), # 이미지의 크기를 224 by 224 로 조정 : 일반적으로 이 크기를 사용
    transforms.CenterCrop(224), # 이미지 중앙을 또 한 번 24 by 224 로 자름 : 중요 부분
    transforms.ToTensor(), # 이미지를 텐서로 변환
    transforms.Normalize( [0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225] ) # 이미지의 채널을 정규화
])
```

**데이터셋을 불러오고 이상이 없는지 확인하는 과정. 가공된 데이터 프레임을 새로 만듦.**


```python
# 전체가 확인하는 코드

data=pd.read_csv('/content/Train')
print(len(data))
class_names=sorted(data['label'].unique().tolist())
print(class_names)
print(len(class_names))
N=list(range(len(class_names)))
normal_mapping=dict(zip(class_names,N))
reverse_mapping=dict(zip(N, class_names))
data['label2']=data['label'].map(normal_mapping)
dir0='/content/'
data['path']=dir0+data['filename']
display(data)
```

**이 함수는 데이터 프레임의 이미지 파일 경로와 레이블을 묶은 튜플을 원소로 가지는 리스트를 생성.**


```python
def create_path_label_list(df):
    path_label_list= []
    for _, row in df.iterrows():
        path=row['path']
        label=row['label2']
        path_label_list.append((path,label))
    return path_label_list

# 확인
#path_label=create_path_label_list(data)
#print(path_label[0:3])
```

**데이터 로더를 위한 클래스 하나를 만듦.**


```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_label, transform=None): # 생성자
        self.path_label = path_label # 방금 위에서 만든 함수의 반환
        self.transform = transform # 아까 위에서 만든 전처리 파이프라인

    def __len__(self):
        return len(self.path_label)

    # 인덱스 사용해 이미지 파일 경로와 레이블 추출, 이미지에 전처리 과정을 적용
    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB') # RGB : Channel=3

        if self.transform is not None:
            img = self.transform(img)

        return img, label
```

**PyTorch Lightning은 PyTorch 간편화 라이브러리인데 이것을 사용해 이미지 데이터셋 로드 후 전처리를 진행.**


```python
class ImageDataset(pl.LightningDataModule):
    def __init__(self, path_label, batch_size=32):
        super().__init__()
        self.path_label = path_label
        self.batch_size = batch_size # 데이터 로더의 반환 배치 크기를 지정

        # 전처리 파이프라인을 새롭게 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # 데이터 모듈을 설정하는 메소드, 데이터 로드 후 훈련용과 테스트용 데이터 분류
    def setup(self, stage=None):
        dataset = CustomDataset(self.path_label, self.transform)
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size

        self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        self.test_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))

    def __len__(self): # 데이터셋의 길이 반환(훈련 or 테스트)
        if self.train_dataset is not None:
            return len(self.train_dataset)
        elif self.test_dataset is not None:
            return len(self.test_dataset)
        else:
            return 0

    # 인덱스를 사용해 하나의 샘플 데이터 반환(훈련 or 테스트)
    def __getitem__(self, index):
        if self.train_dataset is not None:
            return self.train_dataset[index]
        elif self.test_dataset is not None:
            return self.test_dataset[index]
        else:
            raise IndexError("Index out of range. The dataset is empty.")

    def train_dataloader(self): # 훈련용
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self): # 검증용
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self): # 테스트용
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
```

**데이터셋을 훈련용과 테스트용으로 분리.**


```python
class DataModule(pl.LightningDataModule):

    def __init__(self, transform=transform, batch_size=32):
        super().__init__()
        self.root_dir = "/content/"
        self.transform = transform # 위와 동일
        self.batch_size = batch_size # 위와 동일

    # 데이터 모듈을 설정하는 메소드
    def setup(self, stage=None):
        dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
        n_data = len(dataset)
        n_train = int(0.8 * n_data)
        n_test = n_data - n_train

        train_dataset, test_dataset =  random_split(dataset, [n_train, n_test])

        # 훈련용과 테스트용 데이터셋을 데이터 로더로 변환
        self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = DataLoader(test_dataset, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.train_dataset

    def test_dataloader(self):
        return self.test_dataset
```

**CNN 모델인데 PyTorch Lightning을 사용하여 간단한 구현이 가능.**


```python
class ConvolutionalNetwork(LightningModule):

    # 이미지 데이터는 이곳에서 2개의 합성곱층과 3개의 전결합층을 거쳐 출력됨
    def __init__(self): # 중요한 생성자
        super(ConvolutionalNetwork, self).__init__()

        # 합성곱층은 2개
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)

        # 실질적 전결합층은 3개, 각 메소드의 첫 인자는 이전 결과(출력)를 사용
        self.fc1 = nn.Linear(16 * 54 * 54, 120) # 120개 뉴런
        self.fc2 = nn.Linear(120, 84) # 84개 뉴런
        self.fc3 = nn.Linear(84, 20) # 20개 뉴런
        self.fc4 = nn.Linear(20, len(class_names)) # 여긴 softmax 함수 사용해 클래스 분류에 사용됨

    # 순전파 메소드
    def forward(self, X):
        X = F.relu(self.conv1(X)) # 활성화 함수 : ReLu
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X) # 최종 전결합층 거치기
        return F.log_softmax(X, dim=1) # 활성화 함수 : softmax

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) # Adam 옵티마이저
        return optimizer

    # 훈련 단계를 정의한 메소드, 손실값을 반환
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch # 미니 배치
        y_hat = self(X) # 예측된 y
        loss = F.cross_entropy(y_hat, y) # 손실을 구함 : 교차 엔트로피 오차
        pred = y_hat.argmax(dim=1, keepdim=True) # 예측값 : 가장 높은 확률을 가짐
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0] # 정확도
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    # 검증 단계를 정의한 메소드, 알고리즘은 위와 동일
    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    # 테스트 단계를 정의한 메소드, 알고리즘은 위와 동일
    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)
```

**모델을 훈련시키고 테스트. if 구문을 아래와 같이 작성하면 해당 스크립트가 포함된 파일을 외부에서 import 해도 if 구문 내 스크립트는 실행되지 않음.**


```python
if __name__ == '__main__':
    dataset = ImageDataset(path_label) # 데이터셋 변수 생성, 이미지 파일 경로와 레이블들이 튜플로 묶여있는 리스트 path_label

    dataset.setup()  # 데이터를 훈련용과 테스트용으로 분리

    train_dataloader = dataset.train_dataloader() # 훈련용
    test_dataloader = dataset.test_dataloader() # 테스트용

    datamodule = DataModule() # 데이터모듈 객체 생성

    datamodule.setup() # 데이터모듈 : 데이터를 훈련용과 테스트용으로 분리

    model = ConvolutionalNetwork() # CNN 모델 객체 생성

    trainer = pl.Trainer(max_epochs=30) # PyTorch Lightning Trainer : 30 epoch

    trainer.fit(model, datamodule) # CNN 모델 훈련 : 30회

    datamodule.setup(stage='test') # 데이터모듈 : 테스트 모드로 전환

    test_loader = datamodule.test_dataloader() # 데이터모듈 : 테스트용

    trainer.test(dataloaders=test_loader) # 테스트용으로 훈련된 모델을 평가
```

**최종적으로 테스트셋 평가, 분류 결과를 분석.**


```python
device = torch.device("cpu")   # cuda:0 이면 GPU

model.eval()
y_true=[]
y_pred=[]
with torch.no_grad():
    for test_data in datamodule.test_dataloader():
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

print(classification_report(y_true,y_pred,target_names=class_names,digits=4))
```
