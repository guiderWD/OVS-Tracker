import numpy as np
from skimage import io


def convert_to_grid_map(high_res_map, grid_size):
    """
    将细粒度地图转换为降采样的网格地图。
    
    :param high_res_map: 细粒度地图，PIL.Image对象。
    :param grid_size: 网格大小，整数。
    :return: 降采样后的网格地图，PIL.Image对象。
    """
    
    # 获取图片的宽度和高度
    width, height = high_res_map.shape
    
    # 计算降采样后的图片尺寸
    new_width = width // grid_size
    new_height = height // grid_size
    
    # 创建一个新的降采样数组
    grid_map = np.zeros((new_width, new_height), dtype=np.uint8)
    
    # 遍历每个网格
    for i in range(new_height):
        for j in range(new_width):
            # 计算当前网格在原图中的边界
            left = j * grid_size
            top = i * grid_size
            right = (j + 1) * grid_size
            bottom = (i + 1) * grid_size
            
            # 截取当前网格的区域
            area = high_res_map[left:right, top:bottom]
            
            # 获取该区域的最小值
            min_value = np.min(area)
            
            # 如果最小值为0，则将该网格设置为0
            if min_value == 0:
                grid_map[j, i] = 0
            else:
                # 否则，将该网格设置为最小值
                grid_map[j, i] = min_value
    
    return grid_map

# 假设你有一个高分辨率的gridmap
# high_res_gridmap = io.imread(
#     "711_casia.bmp")
# high_res_gridmap = high_res_gridmap.mean(2).astype(np.uint8)
high_res_gridmap = io.imread("711_casia.bmp")
high_res_gridmap = high_res_gridmap.mean(2).astype(np.uint8)

# 定义下采样因子
downsample_factor = 10

# 调用函数进行下采样
low_res_gridmap = convert_to_grid_map(
    high_res_gridmap, downsample_factor)

io.imsave("grid_map_200x100.bmp", low_res_gridmap)
