import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np

# The Goal of the DeDN is to predict the masked region of an image and generate a
# reconstructed version of that image.

def generate_mask_set(image: torch.Tensor, k: int, mask_value = 0) -> torch.Tensor:
    # assuming image shape is [C, H, W]
    assert len(image.shape) == 3
    assert image.shape[-1] % k == 0
    assert image.shape[-2] % k == 0
    
    # Generate mask
    mask = torch.ones(size=(k,k)) * mask_value

    # Create the set of images
    # [batch, C, H, W] where batch = H / K
    masked_image_set = torch.zeros(size=(int(image.shape[-2] / k) * int(image.shape[-1] / k), *image.shape))
    for row in range(int(image.shape[-2] / k)):
        for col in range(int(image.shape[-1] / k)):
            masked_image = image.detach().clone()
            i, j = k * row, k * col 
            masked_image[:, i:i+k, j:j+k] = mask
            masked_image_set[(row*int(image.shape[-2] / k))+col, :, :, :] = masked_image
    
    return masked_image_set

def edge_extraction(image: torch.Tensor, threshold_1: int = 100, threshold_2: int = 200) -> torch.Tensor:
    grayscale_image = torchvision.transforms.Grayscale()(image) # [C, H, W]
    grayscale_image = grayscale_image.permute(-2, -1, 0).numpy()# [H, W, C]
    edge_image = torch.Tensor(cv2.Canny(grayscale_image, threshold_1, threshold_2)).permute(-1, 0, 1)
    return edge_image

def reconstruct_image(patch_images: torch.Tensor, k: int) -> torch.Tensor:
    # assuming image shape is [B, C, H, W]
    assert len(patch_images.shape) == 4
    assert patch_images.shape[-1] % k == 0
    assert patch_images.shape[-2] % k == 0
    
    result_image = torch.zeros(size=(1, *patch_images.shape[-3:]))
    print(result_image.shape)
    
    # Create the set of images
    # [batch, C, H, W] where batch = H / K
    for row in range(int(result_image.shape[-2] / k)):
        for col in range(int(result_image.shape[-1] / k)):
            i, j = k * row, k * col 
            print((row*int(result_image.shape[-2] / k))+col, i, i+k, j, j+k)
            result_image[0, :, i:i+k, j:j+k] = patch_images[(row*int(result_image.shape[-2] / k))+col, :, i:i+k, j:j+k]
    
    return result_image
    
    
if __name__ == "__main__":
    image = torch.randn(size=(3, 10, 10))
    '''result = generate_mask_set(image, k=5)
    print(result.shape)
    print(result[0, ...])
    for i in range(result.shape[0]):
        plt.imshow(result[i, ...].permute(-2, -1, 0))
        plt.show()'''
        
    plt.imshow(image.permute(-2, -1, 0))
    plt.show()
        
    image = image.unsqueeze(0)
    result_patches = torch.concatenate([image] * 4)
    print(result_patches.shape)
    reconstructed_image = reconstruct_image(result_patches, k=5)
    print(reconstructed_image.shape)
    plt.imshow(reconstructed_image[0,  ...].permute(-2, -1, 0))
    plt.show()