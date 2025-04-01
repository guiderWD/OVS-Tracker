import math
import numpy as np
from .utils import Point
from typing import List
def sign(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0

def smooth(v, v0, dt, am):
    a = (v - v0) / dt
    if abs(a) > am and abs(v) > 1e-3:
        v = v0 + am * dt * sign(a)
    return v

def normalize_angle(rad):
    while rad > math.pi:
        rad -= 2 * math.pi
    while rad <= -math.pi:
        rad += 2 * math.pi
    return rad

class VelocityController():
    def __init__(self,
            path: List[Point],
            end_pt: Point,
            watch_on: Point
        ):
        # maximum angular velocity and acceleration (rad)
        self.max_omega = 2.5
        self.max_alpha = 15.0
        # maximum linear velocity (centimeter)
        self.max_vel_x = 200
        self.max_acc_x = 1000
        self.max_vel_y = 200
        self.max_acc_y = 1000
        self.dt = 0.1

        self._path = path
        self.end_pt = end_pt
        self.watch_on_pt = watch_on

        self._angular_vel = 0
        self._linear_vel_x = 0
        self._linear_vel_y = 0
        self._in_flag = False

    @property
    def path(self):
        return self._path

    @property
    def _polygons(self):
        # to `map` frame: x' = 13.35 - x / 100.0; y' = - 5.074 + y / 100.0
        return [
            [Point(155, 332), Point(255, 397), Point(255, 136), Point(558, 93),
             Point(816, 93), Point(660, 352), Point(660, 557), Point(810, 744),
             Point(879, 683), Point(749, 522), Point(749, 381), Point(919, 110),
             Point(1158, 110), Point(1158, 8), Point(155, 8),],
            [Point(1845, 667), Point(1743, 602), Point(1743, 862), Point(1444, 907),
             Point(1181, 907), Point(1339, 644), Point(1339, 441), Point(1189, 254),
             Point(1119, 314), Point(1249, 477), Point(1249, 616), Point(1085, 887),
             Point(842, 887), Point(842, 992), Point(1845, 992),],
        ]

    def _is_point_in_polygon(self, polygon: List[Point], pt: Point) -> bool:
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if pt.y > min(p1y, p2y):
                if pt.y <= max(p1y, p2y):
                    if pt.x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (pt.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or pt.x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


    def _in_region(self, p: int, pt: Point) -> bool:
        foresee_pt = self._find_furthest_point_in_range(
            p, pt.x, pt.y, max_dist=80.0)
        backsee_pt = self._find_furthest_point_in_range(
            len(self.path)-1-p, pt.x, pt.y,
            max_dist=80.0, path=self.path[::-1])
        for polygon in self._polygons:
            if self._is_point_in_polygon(polygon, pt):
                return True
            elif self._is_point_in_polygon(polygon, foresee_pt):
                return True
            elif self._is_point_in_polygon(polygon, backsee_pt):
                return True
        return False


    def naive_velocity_controller(self,
            velocity, watch_on: Point, dists: List[float], states: List[float]):
        dx, dy = dists; x, y, theta = states
        # (angular velocity) = (maximum angular velocity) * (angular difference)
        # tanh: when angular difference = 0, angular velocity = 0
        velocity['angular']['z'] = self.max_omega * np.tanh(normalize_angle(
                math.atan2(watch_on.y - y, watch_on.x - x) - theta))
        if abs(velocity['angular']['z']) < 1e-2:
            velocity['angular']['z'] = 0
        # linear control: free from t1
        sin_yaw0 = math.sin(theta); cos_yaw0 = math.cos(theta)
        norm = np.sqrt(dx ** 2 + dy ** 2)
        velocity['linear']['x'] = (dx * cos_yaw0 + dy * sin_yaw0) / norm * self.max_vel_x
        velocity['linear']['y'] = (dy * cos_yaw0 - dx * sin_yaw0) / norm * self.max_vel_y
        # XXX: intuitive
        ratio = 1 - min(abs(velocity['angular']['z'] / self.max_omega) * 2, 0.85)
        velocity['linear']['x'] *= ratio
        velocity['linear']['y'] *= ratio



    def _find_nearest_point_on_path(self, x, y):
        min_distance = float('inf'); min_p = len(self.path) - 1
        for p, point in enumerate(self.path):
            distance = math.sqrt((point.x - x)**2 + (point.y - y)**2)
            if distance < min_distance:
                min_distance = distance; min_p = p
        return min_p


    def _find_furthest_point_in_range(self, p, x, y, max_dist=50.0, path=None):
        if path is None:
            path = self.path

        if p+1 == len(path):
            #print(path)
            #print(p)
            if p==-1:
                return None
            return path[p]
        for i in range(p+1, len(path)):
            if i == p+1:
                distance = math.sqrt(
                    (path[i].x - x)**2 + (path[i].y - y)**2)
            else:
                distance += math.sqrt(
                    (path[i].x - path[i-1].x)**2 +\
                    (path[i].y - path[i-1].y)**2)
            if distance > max_dist:
                break
        return path[i-1] if i-1 > p else path[i]
        # except (IndexError, TypeError) as e:  # Catch potential errors
        #     print(f"An error occurred in _find_furthest_point_in_range: {e}")
        #     # You might want to log the error more robustly here, e.g., using logging module
        #     return None  # or raise the exception depending on your error handling strategy

    # def _find_furthest_point_in_range(self, p, x, y, max_dist=50.0, path=None):
    #     if path is None:
    #         path = self.path

    #     if len(path) <= p:  # Handle cases where p is out of bounds for path
    #         return None # or raise an exception, depending on your error handling strategy

    #     furthest_point = path[p]  # Initialize with the current point
    #     cumulative_distance = 0.0

    #     for i in range(p + 1, len(path)):
    #         segment_distance = math.sqrt(
    #             (path[i].x - path[i - 1].x)**2 + (path[i].y - path[i - 1].y)**2
    #         )
    #         cumulative_distance += segment_distance

    #         if cumulative_distance > max_dist:
    #             furthest_point = path[i - 1]  # Point before exceeding max_dist
    #             break
    #         else:
    #             furthest_point = path[i] # Update furthest point if within max_dist

    #     return furthest_point


    def _update_in_flag(self, p: int, pt: Point):
        self._in_flag = self._in_region(p, pt)


    def out_of_path_controller(self, velocity, states):
        x, y, theta = states
        velocity['linear']['x'] = velocity['linear']['y'] = 0
        if self._in_flag:
            velocity['angular']['z'] = 0
        else:
            # XXX: different from `self._naive_velocity_controller()`
            # modified tanh to achieve faster turning ?
            velocity['angular']['z'] = self.max_omega * np.tanh(
                3 * normalize_angle(
                    math.atan2(self.watch_on_pt.y - y, self.watch_on_pt.x - x)
                    - theta
                )
            )
        return self.smooth(velocity)


    def smooth(self, velocity):
        velocity['angular']['z'] = smooth(
            velocity['angular']['z'], self._angular_vel, self.dt, self.max_alpha)
        velocity['linear']['x'] = smooth(
            velocity['linear']['x'], self._linear_vel_x, self.dt, self.max_acc_x)
        velocity['linear']['y'] = smooth(
            velocity['linear']['y'], self._linear_vel_y, self.dt, self.max_acc_y)
        self._angular_vel = velocity['angular']['z']
        self._linear_vel_x = velocity['linear']['x']
        self._linear_vel_y = velocity['linear']['y']
        return velocity

    def _is_acute(self, a: Point, b: Point, c: Point):
        ba = Point(a.x - b.x, a.y - b.y)
        bc = Point(c.x - b.x, c.y - b.y)
        dot_product = ba.x * bc.x + ba.y * bc.y
        return dot_product <= 0

    def control(self, x: float, y: float, theta: float):
        velocity = {'linear': {'x': 0, 'y': 0}, 'angular': {'z': 0}}
        p = self._find_nearest_point_on_path(x, y)
        self._update_in_flag(p, Point(x, y))

        # arrive at the target or out of the path
        #!FIXME add an controller for path end (if path[-1] != end)
        if p == len(self.path) - 1:
            return self.out_of_path_controller(velocity, [x, y, theta]), None
        
        current_pt = self.path[p+1]
        # current_pt = self.path[p+1] if self._is_acute(
        #     Point(x, y), self.path[p], self.path[p+1]) else self.path[p+1]
        # dx and dy decide the linear velocity, watch on decide the angular velocity
        dx = current_pt.x - x; dy = current_pt.y - y
        if self._in_flag:
            pt = self._find_furthest_point_in_range(p, x, y)
            # prevent from turning on the narrow path or slope
            watch_on = pt if abs(
                normalize_angle(
                    math.atan2(pt.y - y, pt.x - x) - theta
                )) <= math.pi/2 else Point(2*x - pt.x, 2*y - pt.y)
        else:
            watch_on = self.watch_on_pt
        self.naive_velocity_controller(velocity, watch_on, [dx, dy], [x, y, theta])
        velocity = self.smooth(velocity)

        return velocity, current_pt
