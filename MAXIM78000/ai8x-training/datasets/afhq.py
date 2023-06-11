###################################################################################################
# MemeNet dataloader
# Marco Giordano
# Center for Project Based Learning
# 2022 - ETH Zurich
###################################################################################################
"""
MemeNet dataset
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image

import ai8x

import os
import pandas as pd

import matplotlib.pyplot as plt

"""
Custom image dataset class
"""
class AFHQDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.data_dir = os.path.join('/Users/tristantorchet/Desktop/MA4/MLonMCU2/Exercise8/', img_dir)
        self.transform = transform
        self.classes = os.listdir(self.data_dir)  # Assuming train directory has all classes
        self.image_paths = []
        self.labels = []

        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, class_name)
            image_files = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, img_file) for img_file in image_files])
            self.labels.extend([i] * len(image_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)



        return image, label

"""
Dataloader function
"""
def afhq_get_datasets(data, load_train=False, load_test=False, args=None):
   
    (data_dir, args) = data
    # data_dir = data

    if load_train:
        train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomAffine(degrees=30, translate=(0.5, 0.5), scale=(0.5,1.5), fill=0),
            
            ############################
            # TODO: Add more transform #
            ############################
            
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = AFHQDataset(img_dir=os.path.join(data_dir, "afhq", "train"), transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # # 960 and 720 are not random, but dimension of input test img
            # transforms.CenterCrop((960,720)),
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = AFHQDataset(img_dir=os.path.join(data_dir, "afhq", "val"), transform=test_transform)

        # if args.truncate_testset:
        #     test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'afhq',
        'input': (3, 64, 64),
        'output': list(map(str, range(3))),
        'loader': afhq_get_datasets,
    }
]



# if __name__ == '__main__':
#     # dataset, _ = memes_get_datasets("./data/memes/train/", True)
#     dataloader = DataLoader(memes_get_datasets("./data", load_train=False, load_test=True), batch_size=4,
#                         shuffle=True, num_workers=0)
#
#     fig, ax = plt.subplots(4, 4)
#
#     for i_batch, sample_batched in enumerate(dataloader):
#         print(i_batch, sample_batched[0].size(),
#             sample_batched[1].size())
#
#         # observe 4th batch and stop.
#         if i_batch < 4:
#             for i, img in enumerate(sample_batched[0]):
#                 print(img.shape)
#                 ax[i_batch, i].imshow(img.permute((1,2,0)))
#
#     plt.title('Batch from dataloader')
#     plt.axis('off')
#     plt.ioff()
#     plt.show()
