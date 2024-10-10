import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.cm

model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True, verbose=False).eval()
model_zoe_n = model_zoe_n.to("cuda")

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)

    if value_transform:
        value = value_transform(value)
        
    value = cmapper(value, bytes=True)
    img = value[..., :3]
    img[invalid_mask] = background_color[:3]
    img = img / 255.0
    img = np.power(img, 2.2) * 255.0
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def get_zoe_depth_map(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = torch.tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2).float().to("cuda") / 255.0
    with torch.no_grad():
        depth_map = model_zoe_n.infer_pil(image)
    
    depth_colored = colorize(depth_map, cmap="gray_r")
    return depth_colored

image_path = "img.png"
input_image = Image.open(image_path)
depth_image = get_zoe_depth_map(input_image)

combined_image = Image.new('RGB', (input_image.width + depth_image.width, input_image.height))
combined_image.paste(input_image, (0, 0))
combined_image.paste(depth_image, (input_image.width, 0))

combined_image.show()
