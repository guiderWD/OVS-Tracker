import cv2
import numpy as np

# Shaohui Li, 2024-12-12
import copy
from .utils import Point
pts = [
    # Point(600, 500), Point(500, 600)
]

class CostMap():
    def __init__(self, map_array):
        self.map_array = map_array
        self.scale = 100
        self.hard_expansion = self.scale * 0.30 
        self.soft_expansion = self.scale * 0.25
    
    def map2esdf(self):        
        # Read and process the map
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        map_array = cv2.erode(self.map_array, kernel)
        
        # Hard expansion
        expanded_map_array = cv2.distanceTransform(
            map_array, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        expanded_map_array = np.where(
            expanded_map_array > self.hard_expansion, 255, 0)
        expanded_map_array = expanded_map_array.astype(np.uint8)

        dynamic_map_array = np.zeros_like(self.map_array)
        # Shaohui Li, 2024-12-12
        for pt in pts:
            if expanded_map_array[pt.y, pt.x] == 255:
                dynamic_map_array[pt.y, pt.x] = 10
        dynamic_expansion = cv2.distanceTransform(
            np.where(dynamic_map_array > 0, 0, 255).astype(np.uint8),
            cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dynamic_expansion = np.where(
            dynamic_expansion > self.hard_expansion, 255, 0)
        dynamic_expansion = dynamic_expansion.astype(np.uint8)

        expanded_map_array = np.where(
            (expanded_map_array == 0) | (dynamic_expansion == 0), 0, 255)
        expanded_map_array = expanded_map_array.astype(np.uint8)  

        # Calculate SDF (Signed Distance Field)
        sdf = cv2.distanceTransform(
            expanded_map_array, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sdf1 = cv2.distanceTransform(
            255 - expanded_map_array, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sdf = np.minimum(sdf, self.soft_expansion)
  
        array = np.where(sdf==0, -sdf1, sdf)

        return expanded_map_array, array
