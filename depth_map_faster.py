
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device='cuda')

# load image
image = Image.open('img.png')

# inference
depth = pipe(image)["depth"]

# Create a figure with 2 subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

# Display the depth image
axs[1].imshow(depth, cmap='gray')
axs[1].set_title('Depth Image')
axs[1].axis('off')

# Show the figure
plt.tight_layout()
plt.show()
