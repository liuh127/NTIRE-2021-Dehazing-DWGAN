import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from test_dataset import dehaze_test_dataset
from model import fusion_net
from torchvision.utils import save_image as imwrite
import os
import time
import re
parser = argparse.ArgumentParser(description='Dehaze')
parser.add_argument('--test_dir', type=str, default='./Please load your hazy image path/')
parser.add_argument('--output_dir', type=str, default='./output_result/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
args = parser.parse_args()

test_dir = args.test_dir
test_batch_size = args.test_batch_size
output_dir =args.output_dir
test_dataset = dehaze_test_dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = fusion_net()

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net)
net.load_state_dict(torch.load('./weights/dehaze.pkl'))
# --- Test --- #
with torch.no_grad():
    net.eval()
    start_time = time.time()
    for batch_idx, (hazy_up,hazy_down,name) in enumerate(test_loader):
        hazy_up = hazy_up.to(device)
        hazy_down = hazy_down.to(device)
        frame_out_up = net(hazy_up)
        frame_out_down = net(hazy_down)
        frame_out = (torch.cat([frame_out_up[:, :, 0:600, :].permute(0, 2, 3, 1), frame_out_down[:, :, 552:, :].permute(0, 2, 3, 1)],1)).permute(0, 3, 1, 2)
        if not os.path.exists(output_dir + '/'):
            os.makedirs(output_dir + '/')
        name= re.findall("\d+",str(name))
        imwrite(frame_out, output_dir + '/' + str(name[0])+'.png', range=(0, 1))
test_time = time.time() - start_time
print(test_time)












