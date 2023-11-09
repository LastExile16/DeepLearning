import torch
import torch.nn as nn

def image_size_conv_output(input_dim: int, padding: list, kernel: int, stride: int, layers: int):
    for i in range(layers):
        output_dim = ((input_dim+(2*padding[i])-(kernel))//stride)+1
        print(f"layer_{i+1}: {output_dim}")
        input_dim = output_dim

def image_size_trans_conv_output(input_dim: int, padding: list, kernel: int, stride: int, layers: int):
    for i in range(layers):
        output_dim = (input_dim-1)*stride-2*padding[i]+(kernel)
        print(f"layer_{i+1}: {output_dim}")
        input_dim = output_dim


def check_discriminator(model: nn.Module, image_res: int = 32):
    batch_size = 16
    model_input = torch.randn(batch_size, 3, image_res, image_res)
    assert model(model_input).shape == torch.Size([batch_size, 1]), \
        'Your model should output a single score for each element in the batch'
    print('Congrats, you successfully implemented your discriminator')


def check_generator(model: nn.Module, latent_dim: int, image_res: int = 32):
    batch_size = 16
    model_input = torch.randn(batch_size, latent_dim, 1, 1)
    assert model(model_input).shape == torch.Size([batch_size, 3, image_res, image_res]), \
        'Your model should output a single score for each element in the batch'
    print('Congrats, you successfully implemented your discriminator')
