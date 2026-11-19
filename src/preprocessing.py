from PIL import Image
import numpy as np

def load_image(path):
    with Image.open(path) as image:
        image=image.convert('RGB')      
        return np.array(image)