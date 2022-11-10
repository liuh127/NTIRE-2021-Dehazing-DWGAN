import torch
import time
import argparse
from model import Dehaze, Discriminator
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from torch.utils.data import DataLoader
import os
from utils_test import to_psnr, to_ssim_skimage, to_rmse
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
from perceptual import LossNetwork
# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='DW-GAN Dehaze')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=25, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./check_points')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
train_dataset = os.path.join('/your/path/')

# --- test --- #
test_dataset = os.path.join('/your/path/')
predict_result = args.predict_result
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir = os.path.join('check_points/')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = Dehaze()
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))
DNet = Discriminator()
# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[3000, 5000, 8000], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000, 7000, 8000], gamma=0.5)
# --- Load training data --- #
dataset = dehaze_train_dataset(train_dataset)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
# --- Load testing data --- #
test_dataset = dehaze_test_dataset(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)
MyEnsembleNet = MyEnsembleNet.to(device)
DNet = DNet.to(device)
writer = SummaryWriter()
# --- Load the network weight --- #
try:
    MyEnsembleNet.load_state_dict(torch.load(os.path.join(args.teacher_model, 'best.pkl')))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')
# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()
msssim_loss = msssim
# --- Strat training --- #
iteration = 0
for epoch in range(train_epoch):
    start_time = time.time()
    scheduler_G.step()
    scheduler_D.step()
    MyEnsembleNet.train()
    DNet.train()
    for batch_idx, (hazy, clean) in enumerate(train_loader):
        iteration += 1
        hazy = hazy.to(device)
        clean = clean.to(device)
        output = MyEnsembleNet(hazy)
        DNet.zero_grad()
        real_out = DNet(clean).mean()
        fake_out = DNet(output).mean()
        D_loss = 1 - real_out + fake_out
        D_loss.backward(retain_graph=True)
        adversarial_loss = torch.mean(1 - fake_out)
        MyEnsembleNet.zero_grad()
        adversarial_loss = torch.mean(1 - fake_out)
        smooth_loss_l1 = F.smooth_l1_loss(output, clean)
        perceptual_loss = loss_network(output, clean)
        msssim_loss_ = -msssim_loss(output, clean, normalize=True)
        total_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.0005 * adversarial_loss + 0.2 * msssim_loss_
        total_loss.backward()
        D_optim.step()
        G_optimizer.step()
        writer.add_scalars('training', {'training total loss': total_loss.item()
                                        }, iteration)
        writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1.item(),
                                            'perceptual': perceptual_loss.item(),
                                            'msssim': msssim_loss_.item()
                                            }, iteration)
        writer.add_scalars('GAN_training', {'d_loss': D_loss.item(),
            'd_score': real_out.item(),
            'g_score': fake_out.item()
        }, iteration)
    if epoch % 5 == 0:
        print('we are testing on epoch: ' + str(epoch))
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            rmse_list = []
            MyEnsembleNet.eval()
            for batch_idx, (hazy, clean) in enumerate(test_loader):
                clean = clean.to(device)
                hazy = hazy.to(device)
                frame_out = MyEnsembleNet(hazy)
                if not os.path.exists('output/'):
                    os.makedirs('output/')
                imwrite(frame_out, 'output/' + str(batch_idx) + '.png', range=(0, 1))
                psnr_list.extend(to_psnr(frame_out, clean))
                ssim_list.extend(to_ssim_skimage(frame_out, clean))
                rmse_list.extend(to_rmse(frame_out, clean))
            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            avr_rmse = sum(rmse_list) / len(rmse_list)
            print(epoch, 'dehazed', avr_psnr, avr_ssim, avr_rmse)
            writer.add_scalars('testing', {'testing psnr': avr_psnr, 'testing rmse': avr_rmse,
                                           'testing ssim': avr_ssim
                                           }, epoch)
            torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir, 'epoch' + str(epoch) + '.pkl'))



