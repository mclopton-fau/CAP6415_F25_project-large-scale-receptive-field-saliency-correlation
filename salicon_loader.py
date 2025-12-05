"""
SALICON Dataset Loader
Loads images and human saliency maps from the SALICON dataset.
SALICON: Jiang et al., "SALICON: Saliency in Context" (CVPR 2015)
Dataset: http://salicon.net/challenge-2017/ -> Choose Images (approx 3gb) and Fixation Maps (approx 0.4gb)
"""

import os
import numpy as np
from PIL import Image
import cv2


class SALICONDataset:
    """Loads SALICON images and their corresponding human saliency maps."""
    
    def __init__(self, data_root, split='val'):
        # data_root points to our extracted SALICON folder
        # we're using 'val' split for this project (5000 images)
        self.data_root = data_root
        self.split = split
        
        self.images_dir = os.path.join(data_root, 'images', split)
        self.maps_dir = os.path.join(data_root, 'maps', split)
        
        self.image_files = self._get_image_files()
        
        print(f"Loaded SALICON {split} split: {len(self.image_files)} images")
    
    def _get_image_files(self):
        """Get sorted list of image filenames from the images directory."""
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        return sorted(image_files)
    
    def __len__(self):
        return len(self.image_files)
    
    def get_image_path(self, idx):
        """Get full path to image at index."""
        return os.path.join(self.images_dir, self.image_files[idx])
    
    def get_saliency_map_path(self, idx):
        """Get full path to corresponding saliency map."""
        img_name = self.image_files[idx]
        # Saliency maps have same filename but .png extension
        map_name = os.path.splitext(img_name)[0] + '.png'
        return os.path.join(self.maps_dir, map_name)
    
    def load_image(self, idx):
        """Load image as PIL Image."""
        img_path = self.get_image_path(idx)
        return Image.open(img_path).convert('RGB')
    
    def load_saliency_map(self, idx, normalize=True):
        """
        Load human saliency map (grayscale PNG from SALICON).
        Normalize to [0,1] so values are comparable to Grad-CAM output.
        """
        map_path = self.get_saliency_map_path(idx)
        
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Saliency map not found: {map_path}")
        
        # Load as grayscale
        saliency_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        
        if saliency_map is None:
            raise ValueError(f"Failed to load: {map_path}")
        
        saliency_map = saliency_map.astype(np.float32)
        
        # Normalize to [0,1] range
        if normalize and saliency_map.max() > 0:
            saliency_map = saliency_map / saliency_map.max()
        
        return saliency_map
    
    def get_pair(self, idx):
        """
        Load both image and its human saliency map together.
        This is what we'll use for comparison with model attention.
        """
        image = self.load_image(idx)
        saliency_map = self.load_saliency_map(idx, normalize=True)
        return image, saliency_map