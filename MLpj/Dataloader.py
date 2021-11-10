import os
import torch.utils.data.dataset
from torchvision.io import read_image
import json

DATAPATH = 'D:/DATASETS/fabric_data/'

class CustomImageDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_dir, transform = None, target_transform = None):
        labels_dir = dataset_dir + 'label_json/'
        img_dir = dataset_dir + 'temp/'
        img_target_dir = dataset_dir + 'trgt/' #可能用不到
        self.transform = transform
        self.target_transform = target_transform

        #框框文件名
        folders = os.listdir(labels_dir)
        self.jsFiles = []
        for folder in folders:
            t = os.listdir(labels_dir + folder + '/')
            for f in t:
                self.jsFiles.append(labels_dir + folder + '/' + f)

        folders = os.listdir(img_dir)
        self.imgFiles = []
        for folder in folders:
            t = os.listdir(img_dir + folder + '/')
            for f in t:
                self.imgFiles.append(img_dir + folder + '/' + f)

        folders = os.listdir(img_target_dir)
        self.trgtFiles = []
        for folder in folders:
            t = os.listdir(img_target_dir + folder + '/')
            for f in t:
                self.trgtFiles.append(img_target_dir + folder + '/' + f)

    def __len__(self):
        return len(self.jsFiles)

    def __getitem__(self, idx):
        jsFile = self.jsFiles[idx]
        load_dict = json.load(open(jsFile)) #{'flaw_type': 0, 'bbox': {'x0': 154, 'x1': 246, 'y0': 149, 'y1': 252}}
        label = (load_dict['flaw_type'], load_dict['bbox']['x0'], load_dict['bbox']['x1'], load_dict['bbox']['y0'], load_dict['bbox']['y1'])
        #分布 [559, 353, 387, 39, 553, 21, 157, 24, 210, 11, 1, 154, 15, 385, 711]

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label