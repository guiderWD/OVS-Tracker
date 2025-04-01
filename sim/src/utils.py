import math
import numpy as np
from .bfs import bfs
from typing import Tuple
from typing import List

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'Point ({self.x}, {self.y})'


class State():
    def __init__(self, x, y, theta, dt):
        self.x = x; self.y = y; self.theta = theta
        self.vx = 0; self.vy = 0; self.omega = 0
        self.dt = dt

    def __repr__(self):
        return f"x: {self.x:.4f}, y: {self.y:.4f}, theta: {self.theta:.2f}"

    def update(self, velocity):
        vx, vy = velocity['linear']['x'], velocity['linear']['y']
        omega = velocity['angular']['z']
        self.x = self.x + vx * np.cos(self.theta) * self.dt\
                        - vy * np.sin(self.theta) * self.dt
        self.y = self.y + vx * np.sin(self.theta) * self.dt\
                        + vy * np.cos(self.theta) * self.dt
        self.theta = self.theta + omega * self.dt
        self.vx = vx; self.vy = vy; self.omega = omega


def pow2(n: int) -> int:
    return n * n

def encode(x: int, y: int, cols: int) -> int:
    return y * cols + x
    
def decode(code: int, cols: int) -> Tuple[int, int]:
    x = code % cols
    y = code // cols
    return x, y

def distance2(x1: int, y1: int, x2: int, y2: int) -> int:
    return pow2(x1 - x2) + pow2(y1 - y2)

def F(g, x0: int, y0: int, end: Point):
    return g + math.sqrt(distance2(x0, y0, end.x, end.y))

def FIX(map_array: np.ndarray, coordinate: Point) -> None:
    assert map_array.dtype == np.uint8
    fixed = bfs(0xFF - map_array, coordinate.x, coordinate.y, 0)
    coordinate.x = fixed[0]
    coordinate.y = fixed[1]

def line(image: np.ndarray, p1: Point, p2: Point) -> List[Point]:
    height, width = image.shape[:2]
    
    assert 0 <= p1.x < width and 0 <= p2.x < width
    assert 0 <= p1.y < height and 0 <= p2.y < height
    
    steep = abs(p2.y - p1.y) > abs(p2.x - p1.x)
    if steep:
        p1, p2 = Point(p1.y, p1.x), Point(p2.y, p2.x)

    delta_x = abs(p2.x - p1.x)
    delta_y = abs(p2.y - p1.y)
    error = int(-delta_x / 2.0)
    y = p1.y
    x = p1.x
    points = []

    for _ in range(delta_x + 1):
        points.append(Point(x, y) if not steep else Point(y, x))
        error += delta_y
        if error > 0:
            y += int(p2.y > p1.y) - int(p1.y > p2.y)
            error -= delta_x
        x += int(p2.x > p1.x) - int(p1.x > p2.x)

    filtered_points = []
    for point in points:
        if 0 <= point.x < width and 0 <= point.y < height:
            filtered_points.append(point)

    return filtered_points


def path2arr(path: List[Point]) -> np.ndarray:
    arr = np.zeros([len(path), 2])
    for i, p in enumerate(path):
        arr[i, 0] = p.x; arr[i, 1] = p.y
    return arr

def arr2path(arr: np.ndarray) -> List[Point]:
    path = []
    for i in range(arr.shape[0]):
        path.append(Point(arr[i, 0], arr[i, 1]))
    return path

def load_points(fname: str) -> List[Point]:
    arr = np.load(fname)
    return arr2path(arr)

def save_points(path: List[Point], fname: str):
    arr = path2arr(path)
    np.save(fname, arr)
