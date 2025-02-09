import numpy as np
import math
import dataloaders.transforms as transforms
from dataloaders.dataloader_ext import MyDataloaderExt,Modality,SeqMyDataloaderExt

#iheight, iwidth = 480, 752 # raw image size

class VISIMDataset(MyDataloaderExt):
    def __init__(self, root, type, sparsifier=None, modality='rgb', is_resnet = False,depth_divider=0,max_gt_depth=math.inf):
        super(VISIMDataset, self).__init__(root, type, sparsifier,max_gt_depth, modality)
        self.depth_divider = depth_divider


        if is_resnet:
            self.output_size = (228, 304)
        else:
            self.output_size = (240, 320) # input : (480 , 752)

    def train_transform(self, attrib_list):

        iheight = attrib_list['gt_depth'].shape[0] # 480
        iwidth = attrib_list['gt_depth'].shape[1]  # 752

        s = np.random.uniform(1.0, 1.5) # random scaling

        angle = np.random.uniform(-15.0, 15.0)  # random rotation degrees
        hdo_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        vdo_flip = np.random.uniform(0.0, 1.0) < 0.5  # random vertical flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(270.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(hdo_flip),
            transforms.VerticalFlip(vdo_flip)
        ])

        attrib_np = dict()

        if self.depth_divider == 0:
            if 'fd' in attrib_list:
                minmax_image = transform(attrib_list['fd'])
                max_depth = max(minmax_image.max(),1.0)
            if 'kor' in attrib_list:
                minmax_image = transform(attrib_list['kor'])
                max_depth = max(minmax_image.max(),1.0)
            else:
                max_depth = 50

            scale = 10.0 / max_depth  # 10 is arbitrary. the network only converge in a especific range
        else:
            scale = 1.0 / self.depth_divider

        attrib_np['scale'] = 1.0 / scale

        for key, value in attrib_list.items():
            attrib_np[key] = transform(value)
            if key in Modality.need_divider: #['gt_depth','fd','kor','kde','kgt','dor','dde', 'd3dwde','d3dwor','dvor','dvde','dvgt']:
                attrib_np[key] =  scale*attrib_np[key] #(attrib_np[key] - min_depth+0.01) / (max_depth - min_depth) #/
            elif key in  Modality.image_size_weight_names: #['d2dwor', 'd2dwde', 'd2dwgt']:
                attrib_np[key] = attrib_np[key] / (iwidth * 1.5)  # 1.5 about sqrt(2)- square's diagonal

        if 'rgb' in attrib_np:
            attrib_np['rgb'] = self.color_jitter(attrib_np['rgb'])  # random color jittering
            attrib_np['rgb'] = (np.asfarray(attrib_np['rgb'], dtype='float') / 255).transpose((2, 0, 1))#all channels need to have C x H x W

        if 'grey' in attrib_np:
            attrib_np['grey'] = np.expand_dims(np.asfarray(attrib_np['grey'], dtype='float') / 255, axis=0)

        return attrib_np

    def val_transform(self,  attrib_list):

        iheight = attrib_list['gt_depth'].shape[0] # 480
        iwidth = attrib_list['gt_depth'].shape[1]  # 752

        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight), # Resize(0.5)
            transforms.CenterCrop(self.output_size),
        ])

        attrib_np = dict()

        if self.depth_divider == 0:
            if 'fd' in attrib_list:
                minmax_image = transform(attrib_list['fd'])
                max_depth = max(minmax_image.max(),1.0)
            if 'kor' in attrib_list:
                minmax_image = transform(attrib_list['kor'])
                max_depth = max(minmax_image.max(),1.0)
            else:
                max_depth = 50

            scale = 10.0 / max_depth  # 10 is arbitrary. the network only converge in a especific range
        else:
            scale = 1.0 / self.depth_divider

        attrib_np['scale'] = 1.0 / scale

        for key, value in attrib_list.items():
            attrib_np[key] = transform(value)
            if key in Modality.need_divider:  #['gt_depth','fd','kor','kde','kgt','dor','dde', 'd3dwde','d3dwor','dvor','dvde','dvgt']:
                attrib_np[key] =  scale*attrib_np[key] #(attrib_np[key] - min_depth+0.01) / (max_depth - min_depth)
            elif key in Modality.image_size_weight_names:
                attrib_np[key] = attrib_np[key] / (iwidth*1.5)#1.5 about sqrt(2)- square's diagonal
            elif key == 'rgb':
                attrib_np[key] = (np.asfarray(attrib_np[key], dtype='float') / 255).transpose((2, 0, 1))
            elif key == 'grey':
                attrib_np[key] = np.expand_dims(np.asfarray(attrib_np[key], dtype='float') / 255, axis=0)

        return attrib_np


class VISIMSeqDataset(SeqMyDataloaderExt):
    def __init__(self, root, type, sparsifier=None, modality='rgb', is_resnet = False,depth_divider=1.0,max_gt_depth=math.inf):
        super(VISIMSeqDataset, self).__init__(root, type, sparsifier,max_gt_depth, modality)

        self.depth_divider = depth_divider

        if is_resnet:
            self.output_size = (228, 304)
        else:
            self.output_size = (240, 320)

    def seq_transform(self,attrib_list,is_validation):

        iheight = attrib_list['gt_depth'].shape[0]
        iwidth = attrib_list['gt_depth'].shape[1]

        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),  # this is for computational efficiency,
            transforms.CenterCrop(self.output_size)
        ])

        attrib_np = dict()
        network_max_range = 10.0  # 10 is arbitrary. the network only converge in a especific range
        if 'scale' in attrib_list and attrib_list['scale'] > 0:
            scale = 1.0 / attrib_list['scale']
            attrib_np['scale'] = attrib_list['scale']
        else:
            if 'fd' in attrib_list:
                minmax_image = transform(attrib_list['fd'])
                max_depth = max(minmax_image.max(), 1.0)
            if 'kor' in attrib_list:
                minmax_image = transform(attrib_list['kor'])
                max_depth = max(minmax_image.max(), 1.0)
            else:
                max_depth = 50

            scale = network_max_range / max_depth
            attrib_np['scale'] = 1.0 / scale

        for key, value in attrib_list.items():
            if key not in Modality.no_transform:
                attrib_np[key] = transform(value)
            else:
                attrib_np[key] = value
            if key in Modality.need_divider:
                attrib_np[key] =  scale*attrib_np[key]
            elif key in Modality.image_size_weight_names:
                attrib_np[key] = attrib_np[key] / (iwidth * 1.5)  # 1.5 about sqrt(2)- square's diagonal

        if 'rgb' in attrib_np:
            if not is_validation:
                attrib_np['rgb'] = self.color_jitter(attrib_np['rgb'])  # random color jittering
            attrib_np['rgb'] = (np.asfarray(attrib_np['rgb'], dtype='float') / 255).transpose(
                (2, 0, 1))  # all channels need to have C x H x W

        if 'grey' in attrib_np:
            attrib_np['grey'] = np.expand_dims(np.asfarray(attrib_np['grey'], dtype='float') / 255, axis=0)

        return attrib_np

    def train_transform(self, attrib_list):
        return self.seq_transform(attrib_list, False)

    def val_transform(self, attrib_list):
        return self.seq_transform(attrib_list, True)
