import os
import json
import numpy as np

from .utils import Point

from typing import List

CONNECTION_FILE = "connectivity.txt"
DISTANCE_FILE = "distance.txt"
N2L_FILE = "location_of_points.json"
L2N_FILE = "division_of_points.json"

class Node:
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f'Node {self.id}'

class Grid:
    def __init__(self):
        target_dir = os.path.dirname(__file__)
        target_dir = target_dir.rsplit("/", 1)[0]

        self.connection = np.loadtxt(
            os.path.join(target_dir, "grid", CONNECTION_FILE)
        )
        self.connection[38, 92] = 0; self.connection[92, 38] = 0
        self.distance = np.loadtxt(
            os.path.join(target_dir, "grid", DISTANCE_FILE)
        )
        
        f_pth = os.path.join(target_dir, "grid", N2L_FILE)
        with open(f_pth, "r") as file:
            self.node2location = json.load(file)
        f_pth = os.path.join(target_dir, "grid", L2N_FILE)
        with open(f_pth, "r") as file:
            self.location2node = json.load(file)

        self.num_nodes = len(self.node2location)

    def getNode(self, pt: Point):
        x = int(np.round((pt.x - 5.0) / 10.0))
        y = int(99 - np.round((pt.y - 5.0) / 10.0))
        x = max(min(x, 199), 0)
        y = max(min(y, 99), 0)
        try:
            node = self.location2node[f"{x}, {y}"]
            return Node(node)
        except:
            raise ValueError("Non-exsited location")
    
    def getPoint(self, n: Node) -> Point:
        location = self.node2location[f"{n.id}"]
        location = [location[0] * 10.0 + 5.0, (99 - location[1]) * 10.0 + 5.0 ]
        return Point(*location)

    def pathDist(self, start: Node, goal: Node):
        return self.distance[start.id, goal.id]

    def getNeighbors(self, n: Node) -> List[Node]:
        nodes = np.where(self.connection[n.id] == 1)[0]
        return [Node(nd) for nd in nodes]