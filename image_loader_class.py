from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import random
#import crash_on_ipy

def default_image_loader(path):
    return Image.open(path)

class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, base_path, filenames_filename, n_imgs, ignore_label=255, crop_size=None, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
        """
        self.root = root
        self.base_path = base_path
        self.ignore_label = ignore_label
        self.crop_size = crop_size
        filenamelist = []
        for line in open(os.path.join(self.root, self.base_path, filenames_filename)):
            filenamelist.append((line.split()[0], line.split()[1]))

        np.random.shuffle(filenamelist)

        self.filenamelist = filenamelist[:n_imgs]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, label_path = self.filenamelist[index]
        if os.path.exists(os.path.join(self.root, self.base_path, img_path)) \
                and os.path.exists(os.path.join(self.root, self.base_path, label_path)):
            img = self.loader(os.path.join(self.root, self.base_path, img_path))
            label = self.loader(os.path.join(self.root, self.base_path, label_path))

            if self.crop_size is not None:
                w, h = img.size
                th, tw = self.crop_size

                if w < tw or h < th:  # in case the the original image size is small than the crop size
                    new_im = Image.new('RGB', (tw, th))
                    new_im.paste(img, (0,0))
                    new_im.paste(img, (w, 0))
                    new_im.paste(img, (0, h))
                    new_im.paste(img, (w, h))
                    img = new_im

                    new_lbl = Image.new('L', (tw, th))
                    new_lbl.paste(label, (0, 0))
                    new_lbl.paste(label, (w, 0))
                    new_lbl.paste(label, (0, h))
                    new_lbl.paste(label, (w, h))
                    label = new_lbl
                    w, h = img.size

                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                img = img.crop((x1, y1, x1 + tw, y1 + th))
                label = label.crop((x1, y1, x1 + tw, y1 + th))

            if self.transform is not None:
                try:
                    img = self.transform(img)
                except Exception as e:
                    print(e)

            label = np.array(label, dtype=np.int32)
            label[label == self.ignore_label] = -1
            label = torch.from_numpy(label).long()

            return img, label
        else:
            return None

    def __len__(self):
        return len(self.filenamelist)

