import numpy as np
import heapq
import math
from .utils import FIX, encode, decode, distance2, F, line, Point
from scipy.optimize import minimize
from scipy.optimize import Bounds
from typing import List

class PathSearcher:
    def plan(self, map_array: np.ndarray, start: Point, end: Point) -> List[Point]:
        if not map_array[end.y, end.x]:
            FIX(map_array, end)
        if not map_array[start.y, start.x]:
            FIX(map_array, start)
        path = self.keypoint(self.astar(map_array, start, end))
        return self.dijkstra(self.graph(map_array, path), path) if path else path

    def keypoint(self, path: List[Point]) -> List[Point]:
        keypoints = []
        n = len(path)
        if not n:
            return keypoints
        keypoints.append(path[0])
        #p0, p1, p2 = path[0], path[1], path[2] if n > 2 else path[1]
        try:
            p0, p1, p2 = path[0], path[1], path[2]
        except IndexError:
            if len(path) == 2:
                p0, p1, p2 = path[0], path[1], path[1]  # 或其他合适的处理方式
            elif len(path) == 1:
                print("bbbbbbbbbbbbbbbbbuuuuuuuuuuuuuuuuuuuuuuuugggggggggggggggggggggg")
                p0, p1, p2 = path[0], path[0], path[0]  # 或其他合适的处理方式
            else:
                # 处理 path 为空的情况
                p0, p1, p2 = None, None, None  # 或引发异常
        for i in range(3, n):
            if ((p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x)) != 0:
                keypoints.append(p1)
            p0 = p1; p1 = p2; p2 = path[i]
        keypoints.append(p2)
        return keypoints

    def astar(self, map_array: np.ndarray, start: Point, end: Point) -> List[Point]:
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

    def graph(self, map_array: np.ndarray, points: List[Point]) -> np.ndarray:
        n = len(points)
        g = np.zeros([n, n], dtype=int)
        for i in range(n):
            p1 = points[i]
            for j in range(i + 1, n):
                p2 = points[j]
                connected = True
                if j - i > 1:
                    for pt in line(map_array, p1, p2):
                        if not map_array[pt.y, pt.x]:
                            connected = False
                            break
                if connected:
                    g[i][j] = int(math.sqrt(distance2(p1.x, p1.y, p2.x, p2.y)))
        return g

    def dijkstra(self, g: np.ndarray, points: List[Point]):
        n = g.shape[0]
        dist = [np.inf] * n
        parent = [-1] * n
        visited = [False] * n
        dist[0] = 0
        queue = []
        
        heapq.heappush(queue, (0, 0))
        while queue:
            u = heapq.heappop(queue)[1]
            if u == n - 1:
                path = []
                path.append(points[u])
                u = parent[u]
                while u != 0:
                    path.append(points[u])
                    u = parent[u]
                path.append(points[0])
                return path[::-1]
            for v in range(u + 1, n):
                if not visited[v] and g[u][v] != 0:
                    d = dist[u] + g[u][v]
                    if d < dist[v]:
                        dist[v] = d
                        parent[v] = u
                        heapq.heappush(queue, (d, v))
            visited[u] = True
        return []


class TrajectoryOptimizer:
    def __init__(self, sdf):
        self.t = 0.1
        self.step = 1#30
        self.scale = 100
        self.q = [Point(0, 0) for _ in range(4096)]
        self.max_vel = 1.5 * self.scale
        self.max_acc = 2.5 * self.scale
        self.lambda_f = 1e-4
        self.lambda_s = 2e-8 #2e-8
        self.lambda_d = 1.0
        self.safe = 5.0
        self.sdf = sdf
    
    def plan(self,
            map_array: np.ndarray, 
            path: List[Point]
        ) -> List[Point]:
        if len(path) == 0:
            return path
        points = self.samples(map_array, path)
        points = self.optimize(points)
        return points

    def bspline(self,
            points: List[Point],
            vel_x: float, vel_y: float,
            acc_x: float, acc_y: float
        ) -> List[Point]:
        n = len(points)
        control = [None] * (n + 2)
        p = np.array([1, 4, 1]); v = np.array([-1, 0, 1]); a = np.array([1, -2, 1])
        x = np.zeros(n + 4); y = np.zeros(n + 4)
        matrix = np.zeros((n + 4, n + 2))
        for i in range(n):
            x[i] = points[i].x
            y[i] = points[i].y
            matrix[i, i:i+3] = p / 6
        x[-4] = vel_x; y[-4] = vel_y
        x[-2] = acc_x; y[-2] = acc_y
        x[-3] = y[-3] = x[-1] = y[-1] = 0
        
        matrix[-4, 0:3] = v * 0.5 / self.t
        matrix[-2, 0:3] = a / (self.t * self.t)
        matrix[-3, -3:] = v * 0.5 / self.t
        matrix[-1, -3:] = a / (self.t * self.t)
        cx = np.linalg.lstsq(matrix, x, rcond=None)[0]
        cy = np.linalg.lstsq(matrix, y, rcond=None)[0]
        for i in range(n + 2):
            control[i] = Point(cx[i], cy[i])
        return control

    def samples(self,
            map_array: np.ndarray, keypoints: List[Point]
        ) -> List[Point]:
        points = []
        for i in range(len(keypoints) - 1):
            p1 = keypoints[i]
            p2 = keypoints[i + 1]
            lines = line(map_array, p1, p2)
            for j in range(1, len(lines)):
                if (j-1) % self.step == 0:
                    points.append(lines[j-1])
        if keypoints[-1] not in points:
            points.append(keypoints[-1])
        return points

    def optimize(self, points: List[Point]) -> List[Point]:
        n = len(points)
        x = np.zeros(n * 2)
        for i in range(n):
            x[2 * i] = points[i].x
            x[2 * i + 1] = points[i].y
        upper = np.zeros(n * 2)
        upper[0::2] = self.sdf.shape[1]; upper[1::2] = self.sdf.shape[0]
        lower = np.zeros(n * 2)
        res = minimize(
            self.objective, x, jac=True, method='L-BFGS-B',
            bounds=Bounds(lower, upper),
            options={'maxiter': 10, 'disp': False}
        )
        points_op = [Point(0,0) for _ in range(n)]
        for i in range(n):
            points_op[i].x = res.x[2 * i]
            points_op[i].y = res.x[2 * i + 1]
        return points_op

    def objective(self, x):
        n = len(x) // 2
        cost = 0.0
        grad = np.zeros_like(x)
        gradient = np.zeros((n, 2))
        
        # Update control points
        for i in range(n):
            idx = i * 2
            self.q[i].x = x[idx]
            self.q[i].y = x[idx + 1]
        
        # Calculate velocities, accelerations, and jerks
        vel = np.array([
            (self.q[i + 1].x - self.q[i].x, self.q[i + 1].y - self.q[i].y)
            for i in range(n - 1)])
        acc = np.array([
            (vel[i + 1][0] - vel[i][0], vel[i + 1][1] - vel[i][1])
            for i in range(n - 2)])
        jerk = np.array([
            (acc[i + 1][0] - acc[i][0], acc[i + 1][1] - acc[i][1])
            for i in range(n - 3)])
        
        # Feasibility Cost
        vm2 = (self.max_vel) ** 2
        am2 = (self.max_acc) ** 2
        t2 = 1 / self.t / self.t
        t4 = t2 ** 2
        for i in range(1, n - 2):
            for z in range(2):
                cost_velocity = (vel[i][z] ** 2) * t2 - vm2
                if cost_velocity > 0:
                    cost += self.lambda_f * cost_velocity
                    value = 2 * self.lambda_f * vel[i][z] * t2
                    gradient[i][z] -= value
                    gradient[i+1][z] += value
        for i in range(1, n - 3):
            for z in range(2):
                cost_acceleration = (acc[i][z] ** 2) * t4 - am2
                if cost_acceleration > 0:
                    cost += self.lambda_f * cost_acceleration
                    value = 2 * self.lambda_f * acc[i][z] * t4
                    gradient[i][z] += value
                    gradient[i+2][z] += value
                    gradient[i+1][z] -= 2 * value

        # Smoothness Cost
        t6 = t2 * t4
        for i in range(1, n - 4):
            for z in range(2):
                value = 2 * self.lambda_s * jerk[i][z] * t6
                cost += self.lambda_s * t6 * ((jerk[i][z])**2)
                gradient[i][z] -= value
                gradient[i+3][z] += value
                gradient[i+1][z] += 3 * value
                gradient[i+2][z] -= 3 * value
        
        # Safety Cost
        for i in range(1, n-1): # range(3, n + 3):
            x_idx = int(round(self.q[i].x))
            y_idx = int(round(self.q[i].y))
            x_idx = max(1, min(x_idx, self.sdf.shape[1] - 2))
            y_idx = max(1, min(y_idx, self.sdf.shape[0] - 2))
            # out of obstacle
            if self.sdf[y_idx, x_idx] >= 0:
                sqr = np.sqrt(self.sdf[y_idx, x_idx])
                cost_safety = self.safe - sqr
            # in obstacle
            else:
                sqr = np.sqrt(-self.sdf[y_idx, x_idx])
                cost_safety = self.safe + sqr
            if cost_safety >= 0:
                g = self.lambda_d * cost_safety
                cost += g
                gradient[i][0] += g * (
                    self.sdf[y_idx, x_idx - 1] - self.sdf[y_idx, x_idx + 1])
                gradient[i][1] += g * (
                    self.sdf[y_idx - 1, x_idx] - self.sdf[y_idx + 1, x_idx])
                # New
                gradient[i][0] += 0.707 * g * (
                    self.sdf[y_idx - 1, x_idx-1] - self.sdf[y_idx + 1, x_idx+1])
                gradient[i][1] += 0.707 * g * (
                    self.sdf[y_idx - 1, x_idx-1] - self.sdf[y_idx + 1, x_idx+1])
                gradient[i][0] += 0.707 * g * (
                    self.sdf[y_idx + 1, x_idx-1] - self.sdf[y_idx - 1, x_idx+1])
                gradient[i][1] -= 0.707 * g * (
                    self.sdf[y_idx + 1, x_idx-1] - self.sdf[y_idx - 1, x_idx+1])
                
        for i in range(n):
            z = 2 * i
            grad[z] = gradient[i][0]
            grad[z + 1] = gradient[i][1]

        return cost, grad
