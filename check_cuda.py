import torch
import torchvision

""" Checks if cuda is enabled on our device, and if pytorch and torchvision are the same version
    Only needed for training """
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.__version__)
print(torchvision.__version__)