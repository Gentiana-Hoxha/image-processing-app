# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:16:37 2024

@author: STORM
"""

import numpy as np
import matplotlib.pyplot as plt
 
def gaussian_kernel(kernel_size, sigma):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    x, y = np.meshgrid(ax, ax)
    
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    return kernel

kernel = gaussian_kernel(5, 1)
print(kernel)

gaussian_matrix = gaussian_kernel(20, 10)

plt.imshow(gaussian_matrix, cmap='viridis')
plt.title('Gaussian Kernel Matrix')
plt.colorbar()
plt.show()
