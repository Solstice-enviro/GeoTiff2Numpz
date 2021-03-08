import numpy as np
from numpy import save
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

data = np.load('D:/Programming/EuroSat/data/Testing_128x128/npz')
x, y = data[1]


#save('D:/Programming/EuroSat/data/Testing_128x128/npy/test_000.npy', data)
#transform = transforms.Compose([
#    transforms.RandomResizedCrop(224),
#    transforms.ToTensor(),
#])
#dataset = data(transform=transform)
#data = torch.from_numpy(data).unsqueeze(1).float()
#print(data.item())
#b.np.save('D:/Programming/EuroSat/data/Testing_128x128/npy')