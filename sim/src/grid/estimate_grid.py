import os
import json
import numpy as np
import heapq
import tqdm
from skimage import io

from src.utils import Point, encode, decode, F
from src.costmap import CostMap
import matplotlib.pyplot as plt

def astar(map_array: np.ndarray, start: Point, end: Point) -> list[Point]:
    cols = map_array.shape[1]
    rows = map_array.shape[0]
    g = np.full((rows * cols,), np.inf)
    parent = np.full((rows * cols,), -1, dtype=int)
    visited = np.full((rows * cols,), False, dtype=bool)
    queue = []
    
    # start point
    index = encode(start.x, start.y, cols)
    g[index] = 0
    visited[index] = True
    heapq.heappush(queue, (0, index))
    
    while queue:
        index = heapq.heappop(queue)[1]
        visited[index] = True
        
        if index == encode(end.x, end.y, cols):
            path = [Point(x, y) for x, y in [decode(index, cols)]]
            index = parent[index]
            while index != -1:
                path.append(Point(*decode(index, cols)))
                index = parent[index]
            return path[::-1]      
        x, y = decode(index, cols)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < cols) or not (0 <= ny < rows):
                    continue
                idx = encode(nx, ny, cols)
                if visited[idx] or not map_array[ny, nx]:
                    continue
                # origin: cost = F(g[index], nx, ny, end)
                cost = F(g[index], nx, ny, Point(x, y))
                if cost < g[idx]:
                    g[idx] = cost
                    parent[idx] = index
                    heapq.heappush(queue, (F(cost, nx, ny, end), idx))
    return []

N2L_FILE = "location_of_points.json"

target_dir = os.path.dirname(__file__)    
f_pth = os.path.join(target_dir, "grid", N2L_FILE)
with open(f_pth, "r") as file:
    node2location = json.load(file)

map_array = io.imread("map/grid_map_200x100.bmp")
map_array = map_array.astype(np.uint8)
# expanded_map_array, esdf = CostMap(map_array).map2esdf()

distance = np.zeros([100, 100])
distance[:, :] = np.inf
for i in tqdm.tqdm(range(100), desc='ALL'):
    for j in tqdm.tqdm(range(100), desc='minor'):
        start = node2location[f"{i}"]
        end = node2location[f"{j}"]
        path = astar(
            map_array,
            Point(start[0], 99-start[1]),
            Point(end[0], 99-end[1])
        )
        if len(path) == 1:
            distance[i, j] = 0
        else:
            dist = 0
            for k in range(len(path) - 1):
                dist += np.sqrt(
                    (path[k+1].x - path[k].x) ** 2 + 
                    (path[k+1].y - path[k].y) ** 2
                )
            distance[i, j] = dist
np.savetxt('distance.txt', distance, fmt='%f')
