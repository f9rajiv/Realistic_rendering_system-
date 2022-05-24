from django.shortcuts import render

# Create your views here.
import cv2
import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from PIL import Image as im
from pathlib import Path
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import moviepy.video.io.ImageSequenceClip


from django.core.files.storage import FileSystemStorage



SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'test':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)
#
def make_dataloaders(batch_size=95, n_workers=0, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    if file_extension == ".mp4":
        for i in range(95):
            plt.imsave(f'D:/Major project/GAN/rendering_system/media/output/{str(i)}.png',fake_imgs[i])
    else:
        for i in range(1):
            plt.imsave(f'D:/Major project/GAN/rendering_system/media/output/{str(i)}.png',fake_imgs[i])



def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def about(request):
    return render(request,'about.html')

def wireframe(request):
    print(request)
    print(request.POST.dict())

    fileobj=request.FILES['filePath']
    fs=FileSystemStorage()
    global filepathname
    filepathname=fs.save(fileobj.name,fileobj)
    filepathname=fs.url(filepathname)
    print((filepathname))
    dir = 'D:/Major project/GAN/rendering_system/media/output'
    # D:/Major project/GAN/rendering_system/media/output
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
        
    global testimage
    testimage='.'+filepathname
    global file_extension
    filename,file_extension = os.path.splitext(testimage)
    print(file_extension)
    global data
    global ls,abs_
    if file_extension == ".mp4":
        cap= cv2.VideoCapture(testimage)
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite("D:/Major project/GAN/rendering_system/media/test204/"+str(i)+'.jpg',frame)
            i+=1
        # D:/Major project/GAN/rendering_system/media/test204
        path = "D:/Major project/GAN/rendering_system/media/test204"
        paths = glob.glob(path + "/*.jpg")
    
    else:
        paths = "D:/Major project/GAN/rendering_system/media/new_test"
        paths = glob.glob(paths + "/*.jpg")  


    test_path=np.asarray(paths)   
    test_dl = make_dataloaders(paths=test_path, split='test')
    
    data = next(iter(test_dl))
    ls, abs_ = data['L'],data['ab']
    context={'filepathname':filepathname}
    return render(request,'wireframe.html',context)


def viewcode(request):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load("./model/model_e_40.pt",map_location=device)
    visualize(model, data, save=True)
    dir = 'D:/Major project/GAN/rendering_system/media/test204'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


    if file_extension == ".mp4":

        image_folder='D:/Major project/GAN/rendering_system/media/output'
        fps=24
        image_files = [os.path.join(image_folder,img)
                    for img in os.listdir(image_folder)
                    if img.endswith(".png")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile('D:/Major project/GAN/rendering_system/media/output_video/video.mp4')
        pathname="/media/output_video/video.mp4" 
        context={'filepathname':filepathname,'pathname':pathname}
    else:
        pathname="/media/output/0.png"
        context={'filepathname':filepathname,'pathname':pathname}
    return render(request,'viewcode.html',context)


#
