import rembg
import numpy as np
from PIL import Image

# Load the input image
input_image = Image.open('lettuce9dis.jpg')

# Convert the input image to a numpy array
input_array = np.array(input_image)

# Apply background removal using rembg
output_array = rembg.remove(input_array)

# Create a PIL Image from the output array
output_image = Image.fromarray(output_array)
output_image = output_image.convert('RGB')
# Save the output image
output_image.save('marul9.jpg')
