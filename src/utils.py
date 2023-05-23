
import torch
import torch.nn.functional as F

def load_model(m, p, device):
    print(p)
    dict = torch.load(p, map_location=device)
    for i, k in zip(m.state_dict(), dict):
        weight = dict[k]
        m.state_dict()[i].copy_(weight)


# refer : https://github.com/OniroAI/MonoDepth-PyTorch
def warp_feature_using_disparity(feature, disp):
    B, _, H, W = feature.shape

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, W).repeat(B, H, 1).to(dtype=feature.dtype, device=feature.device)
    y_base = torch.linspace(0, 1, H).repeat(B, W, 1).transpose(1, 2).to(dtype=feature.dtype, device=feature.device)

    # Apply shift in X direction
    x_shifts = (x_base * W - disp) / W
    flow_field = torch.stack((x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(feature, 2*flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output
