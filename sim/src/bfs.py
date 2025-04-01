import numpy as np
import queue as queue_lib
from typing import Tuple

def bfs(map_array: np.ndarray, xs: int, ys: int, d_thr: int) -> Tuple[int, int]:
    assert map_array.dtype == np.uint8
    xy = (0, 0)
    x0, y0 = 0, 0
    queue = queue_lib.Queue()
    h, w = map_array.shape[:2]
    xs = max(0, min(xs, w - 1))
    ys = max(0, min(ys, h - 1))
    dist = np.full((h, w), np.iinfo(np.uint32).max, dtype=np.uint32)
    visited = np.full((h, w), False, dtype=bool)
    visited[ys][xs] = True
    dist[ys][xs] = 0
    queue.put((xs, ys))
    while not queue.empty():
        xy = queue.get()
        x0, y0 = xy
        if dist[y0][x0] > d_thr:
            break
        for dxy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            dx, dy = dxy
            x = x0 + dx
            if x < 0 or x >= w:
                continue
            y = y0 + dy
            if y < 0 or y >= h:
                continue
            d = dist[y0][x0] + np.sqrt(dx**2 + dy**2)
            if not visited[y][x]:
                visited[y][x] = True
                queue.put((x, y))
                dist[y][x] = 0 if map_array[y, x] > 0 else d
            else:
                dist[y][x] = min(dist[y][x], d)
    return xy
