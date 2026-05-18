#######################################################################
# Implementation of 
# A Sliced Wasserstein Loss for Neural Texture Synthesis
# Heitz et al., CVPR 2021
#######################################################################

import numpy as np
import torch
import torch.nn as nn
from platformdirs import user_data_dir

from SWLoss.src.get_and_load_pth import get_and_load_pth


class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsampling = torch.nn.AvgPool2d((2,2))

    def forward(self, image):
        
        # RGB to BGR
        image = image[:, [2,1,0], :, :]

        # [0, 1] --> [0, 255]
        image = 255 * image

        # remove average color
        image[:,0,:,:] -= 103.939
        image[:,1,:,:] -= 116.779
        image[:,2,:,:] -= 123.68

        # block1
        block1_conv1 = self.relu(self.block1_conv1(image))
        block1_conv2 = self.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.downsampling(block1_conv2)

        # block2
        block2_conv1 = self.relu(self.block2_conv1(block1_pool))
        block2_conv2 = self.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.downsampling(block2_conv2)

        # block3
        block3_conv1 = self.relu(self.block3_conv1(block2_pool))
        block3_conv2 = self.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = self.relu(self.block3_conv3(block3_conv2))
        block3_conv4 = self.relu(self.block3_conv4(block3_conv3))
        block3_pool = self.downsampling(block3_conv4)

        # block4
        block4_conv1 = self.relu(self.block4_conv1(block3_pool))
        block4_conv2 = self.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = self.relu(self.block4_conv3(block4_conv2))
        block4_conv4 = self.relu(self.block4_conv4(block4_conv3))

        return [block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]

class SlicingLoss(nn.Module):
    def __init__(self, scaling_factor, vgg_model=None):
        super(SlicingLoss, self).__init__()
        if vgg_model is None:
            vgg_model = VGG19()
            state_dict = get_and_load_pth("https://github.com/rymuelle/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis/releases/download/V0.1.0/vgg19.pth")
            vgg_model.load_state_dict(state_dict)
        self.vgg = vgg_model
        self.scaling_factor = scaling_factor
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()

    def forward(self, image_generated, image_example):
        list_activations_generated = self.vgg(image_generated)
        list_activations_example = self.vgg(image_example)
        
        loss = 0.0
        device = image_generated.device
        repeat_factor = self.scaling_factor * self.scaling_factor
        
        # Iterate over layers
        for l in range(len(list_activations_example)):
            # Get dimensions
            b = list_activations_example[l].shape[0]
            dim = list_activations_example[l].shape[1]
            n = list_activations_example[l].shape[2] * list_activations_example[l].shape[3]
            
            # Linearize layer activations and duplicate example activations according to scaling factor
            activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, repeat_factor)
            activations_generated = list_activations_generated[l].view(b, dim, n * repeat_factor)
            
            # Sample random directions on the correct device
            n_direction = dim
            directions = torch.randn(n_direction, dim, device=device)
            directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
            
            # Project activations over random directions
            projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
            projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
            
            # Sort the projections
            sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
            sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
            
            # L2 over sorted lists
            loss += torch.mean((sorted_activations_example - sorted_activations_generated) ** 2)
            
        return loss
    
if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256).to(torch.device("cuda:0"))
    y = torch.rand(1, 3, 256, 256).to(torch.device("cuda:0"))
    loss_fn = SlicingLoss(1).to('cuda')
    print(loss_fn(x,y))
