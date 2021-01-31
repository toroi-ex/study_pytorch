import numpy as np
import matplotlib.pyplot as plt

import torch

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import classification_report

from pathlib import Path

from PIL import Image

from vgg16_bn.model import vgg16_bn_test


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imageSize, dir_path, transform=None):
        self.transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.image_paths = [str(p) for p in Path(dir_path).glob("**/*.png")]

        self.data_num = len(self.image_paths)
        self.classes = ['broken', 'correct']
        self.class_to_idx = {'broken': 1, 'correct': 0}

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        image = Image.open(p).convert('RGB') #画像をRGBに変換することで、3チャネルにする

        if self.transform:
            out_data = self.transform(image)

        out_label = p.split("/") #フォルダの名前を”/”で分割
        out_label = self.class_to_idx[out_label[-2]] #後ろから2番めの値を取ってくる（”correct or broken”）

        return out_data, out_label

test_data_dir = '/home/toui/デスクトップ/ori/add_data2/DATA3_test/test200'

model_path = "model.pth"
device = torch.device("cuda:0")

data_set = MyDataset(200, dir_path=test_data_dir)
test_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_img():

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # 画像の表示
    imshow(torchvision.utils.make_grid(images))
    # ラベルの表示
    print(' '.join('%5s' % labels for j in range(4)))

if __name__ == "__main__":

    # net = myVGG()
    # net = models.vgg16(pretrained=False)
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net = vgg16_bn_test()
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net = net.eval()

    pred = []
    Y = []
    target_name = ["correct 0", "broken 1"]
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = net(inputs)
        pred += [int(l.argmax()) for l in output]
        Y += [int(l) for l in labels]

    print(classification_report(Y, pred, target_names=target_name))
    #
    # nb_classes = 2
    #
    # # Initialize the prediction and label lists(tensors)
    # predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    # lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    #
    # with torch.no_grad():
    #     for i, (inputs, classes) in enumerate(test_loader):
    #         inputs = inputs.to(device)
    #         classes = classes.to(device)
    #         outputs = net(inputs)
    #         _, preds = torch.max(outputs, 1)
    #
    #         # Append batch prediction results
    #         predlist=torch.cat([predlist,preds.view(-1).cpu()])
    #         lbllist=torch.cat([lbllist,classes.view(-1).cpu()])
    #
    # # Confusion matrix
    # conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    # print(conf_mat)