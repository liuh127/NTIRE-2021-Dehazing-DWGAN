from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class dehaze_test_dataset(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test=[]
        for i in os.listdir(test_dir):
            self.list_test.append(i)
        self.root_hazy = os.path.join(test_dir)
        self.file_len = len(self.list_test)
    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy + self.list_test[index])
        hazy = self.transform(hazy)
        hazy_up=hazy[:,0:1152,:]
        hazy_down=hazy[:,48:1200,:]
        name=self.list_test[index]
        return hazy_up,hazy_down,name
    def __len__(self):
        return self.file_len





