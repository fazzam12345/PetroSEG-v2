import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import os
from skimage import segmentation


def perform_custom_segmentation(image, params):
    class Args(object):
        def __init__(self, params):
            self.train_epoch = params.get('train_epoch', 2 ** 3)
            self.mod_dim1 = params.get('mod_dim1', 64)
            self.mod_dim2 = params.get('mod_dim2', 32)
            self.gpu_id = params.get('gpu_id', 0)
            self.min_label_num = params.get('min_label_num', 6)
            self.max_label_num = params.get('max_label_num', 256)

    args = Args(params)

    class MyNet(nn.Module):
        def __init__(self, inp_dim, mod_dim1, mod_dim2):
            super(MyNet, self).__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mod_dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mod_dim2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mod_dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mod_dim2),
            )

        def forward(self, x):
            return self.seq(x)

    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=15, sigma=0.06, min_size=14)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    progress_bar = st.progress(0)

    for batch_idx in range(args.train_epoch):
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)

        progress = (batch_idx + 1) / args.train_epoch
        progress_bar.progress(progress)

    return show