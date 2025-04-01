import numpy as np
import cv2
import heapq

from .utils import encode, decode, F, FIX
from .utils import Point, State, line
from typing import List
def astar(map_array: np.ndarray, start: Point, end: Point) -> List[Point]:
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


class WayPointOptimizer():
    def __init__(self, expanded_map_array: np.ndarray, waypoints: List[Point]):
        self.expanded_map_array = expanded_map_array
        self.emap = [None for _ in range(len(waypoints))]
        self.waypoints = waypoints
        # expansions for costmap and sdfs
        self.hard_expansion = 60
        self.soft_expansion = 25
        # weights for searching-based solver
        self.lambda_d = 1.0
        self.lambda_s = 2.0

    def get_emap(self, pts: List[Point]):
        dynamic_map_array = np.zeros_like(self.expanded_map_array)
        for pt in pts:
            if self.expanded_map_array[pt.y, pt.x] == 255:
                dynamic_map_array[pt.y, pt.x] = 10
        dynamic_expansion = cv2.distanceTransform(
            np.where(dynamic_map_array > 0, 0, 255).astype(np.uint8),
            cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dynamic_expansion = np.where(
            dynamic_expansion > self.hard_expansion, 255, 0)
        dynamic_expansion = dynamic_expansion.astype(np.uint8)

        expanded_map_array = np.where(
            (self.expanded_map_array == 0) | (dynamic_expansion == 0), 0, 255)
        expanded_map_array = expanded_map_array.astype(np.uint8)

        sdf = cv2.distanceTransform(
            expanded_map_array, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sdf1 = cv2.distanceTransform(
            255 - expanded_map_array, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sdf = np.minimum(sdf, self.soft_expansion)
  
        array = np.where(sdf==0, -sdf1, sdf)
        
        return expanded_map_array, array

    def _update(self, emap: np.ndarray, esdf: np.ndarray, waypoint: Point, current: Point) -> Point:
        if emap[current.y, current.x] == 0:
            FIX(emap, current)
            
        path = astar(emap, current, waypoint)  #self.expanded_map_array
        if len(path) == 0:
            return current
        all_dist = 0
        for i in range(len(path) - 1):
            all_dist += np.sqrt(
                (path[i].x - path[i+1].x) ** 2 +
                (path[i].y - path[i+1].y) ** 2
            )
        min_cost = np.inf; dist = 0; old_p = path[0]
        for p in path:
            dist += np.sqrt(
                (p.x - old_p.x) ** 2 + (p.y - old_p.y) ** 2
            )
            cost = self._get_cost(esdf, all_dist - dist, p)
            if cost == np.inf:
                break
            if cost < min_cost:
                min_cost = cost
                sub_goal = p
            old_p = p
        return sub_goal

    def _get_cost(self, esdf: np.ndarray, dist: float, p: Point) -> float:
        cost = 0.
        cost += self.lambda_d * dist
        if esdf[p.y, p.x] < 0:
            cost = np.inf
        else:
            cost += self.lambda_s * (self.soft_expansion - esdf[p.y, p.x])
        return cost

    def update(self, states: List[State], dt=0.2):
        waypoints = []
        for i in range(len(self.waypoints)):
            pts = []
            teammate_states = states[:i] + states[i+1:]
            for state in teammate_states:
                waypoint = self.waypoints[i]
                start = Point(int(state.x), int(state.y))
                end = Point(
                    int(np.round(state.x + state.vx * dt)),
                    int(np.round(state.y + state.vy * dt))
                )
                pts += line(self.expanded_map_array, start, end)
            
            emap, esdf = self.get_emap(pts)
            self.emap[i] = emap
            waypoint = self._update(
                emap, esdf, waypoint,
                Point(int(states[i].x), int(states[i].y))
            )
            waypoints.append(waypoint)
        return waypoints