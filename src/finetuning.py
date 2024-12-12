import torch
from torch import nn

from peft import LoraConfig, get_peft_model





def modify_unet_for_multi_frame(unet, num_frames):
    # Modify the first convolution layer
    in_channels = unet.config.in_channels
    unet.config.in_channels = in_channels * num_frames

    old_conv = unet.conv_in
    new_conv = nn.Conv2d(in_channels * num_frames, old_conv.out_channels, 
                         kernel_size=old_conv.kernel_size, 
                         stride=old_conv.stride, 
                         padding=old_conv.padding)
    
    # Initialize the new conv layer with weights from the old layer
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, -in_channels:] = old_conv.weight
        new_conv.bias = old_conv.bias
    
    new_conv.requires_grad_(True)

    # Replace the old conv layer with the new one
    unet.conv_in = new_conv
    return unet

def load_conv_in_weights(unet, path):
    path = f"{path}/conv_in_weights.pth"
    unet.conv_in.load_state_dict(torch.load(path))

def save_conv_in_weights(unet, path):
    torch.save(unet.conv_in.state_dict(), f"{path}/conv_in_weights.pth")

def lora_unet_for_multi_frame(unet, state_size, rank):
    # Without LoRa for now
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.model = modify_unet_for_multi_frame(unet.model, state_size)

    return unet