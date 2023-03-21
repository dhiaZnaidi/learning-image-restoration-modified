import matplotlib.pyplot as plt

import torch
import argparse
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from data_salty import BerkeleyLoaderSaltAndPepper
from model_salty import DiffusionNetSalt

import numpy as np

from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('--steps', '-T', type=int, default=8, help='Number of steps of the net')
    parser.add_argument('--n_filters', '-nf', type=int, default=48, help='Number of filters')
    parser.add_argument('--filter_size', '-fs', type=int, default=7, help='Size of the filters')
    parser.add_argument('model_path', help='Path to the model')
    args = parser.parse_args()
    return args

def calc_psnr(im_true, im):
    im_true = im_true.detach().cpu().numpy()[0].transpose((1,2,0))
    im = im.detach().cpu().numpy()[0].transpose((1,2,0))
    m = psnr(im_true, im)
    return m


def test(model):
        model.eval()
        test_loader = BerkeleyLoaderSaltAndPepper(train=False, num_workers=2)
        test_batches=[]
        psnr_steps = [0] * len(model.dnets)
        psnr_steps_plot = [0] * len(model.dnets)
        for _im_noisy, _im in tqdm(test_loader):
            im_noisy, im = _im_noisy.cuda(), _im.cuda()
            im_pred = torch.clone(im_noisy)
            for i in range(len(model.dnets)):
                with torch.no_grad():
                    im_pred = model.step(im_pred, im_noisy, i)
                psnr_steps[i] += calc_psnr(im, im_pred)
            test_batches.append([_im_noisy,_im,im_pred])
        print(type(im_noisy),type(im),type(im_pred))
        print("Avrg PSNR")
        
        for i in range(len(psnr_steps)):
            print(i+1, "{:.2f} dB".format(psnr_steps[i]/len(test_loader)))
            psnr_steps_plot[i] = psnr_steps[i]/len(test_loader)
        return test_batches,psnr_steps_plot


if __name__ == '__main__':
    args = parse_args()
    model = DiffusionNetSalt(T=args.steps, n_rbf=63, n_channels=1, n_filters=args.n_filters, filter_size=args.filter_size).cuda()
    model.load_state_dict(torch.load(args.model_path))
    test_images,psnrs = test(model)
    i=0
    to_PIL = transforms.ToPILImage()

    stages = np.arange(1,len(psnrs)+1)
    plt.scatter(stages,psnrs,label="PSNR")
    #plt.plot(xs,val_losses,label="Validation Loss")
    plt.title('Salt and Pepper Noise Testing')
    plt.xlabel('Stages')
    plt.ylabel('Average PSNR')
    plt.legend()
    plt.savefig('/content/avg_psnr.png')

    for i in [0,-1,2]:
      im_noisy,im,im_pred = test_images[i]
      fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
      ax1.set_title("Noisy Image")
      ax1.axis("off")
      ax1.imshow(im_noisy[0].permute(1,2,0),cmap = 'gray')
      ax2.set_title("Original Image")
      ax2.axis("off")
      ax2.imshow(im[0].permute(1,2,0),cmap="gray")
      ax3.set_title("Restored Image")
      ax3.axis("off")
      # Convert PyTorch tensor to numpy array
      im_pred_np = im_pred.cpu().detach().numpy().transpose(0, 2, 3, 1)

  # Select the first image from the batch
      im_pred_np = im_pred_np[0]

      ax3.imshow(im_pred_np,cmap="gray")
      fig.savefig('/content/testing_'+str(i)+'.png')
