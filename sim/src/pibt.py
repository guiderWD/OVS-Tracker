import random
import time
import heapq
import numpy as np

from .grid import Grid, Node
from .utils import Point as PointProto
from typing import List

class Point(PointProto):
    def __lt__(self, other):
        if self.x != other.x:
            return self.x > other.x
        if self.y != other.y:
            return self.y > other.y


class Agent:
    def __init__(self, id, current, goal, elapsed=0, dist=0.0, tie_breaker=0.0):
        self.id = id
        self.v_now = current
        self.v_next = None
        self.g = goal
        self.elapsed = elapsed
        self.init_d = dist
        self.tie_breaker = tie_breaker

    def __lt__(self, other) -> bool:
        if self.elapsed != other.elapsed:
            return self.elapsed > other.elapsed
        if self.init_d != other.init_d:
            return self.init_d > other.init_d
        return self.tie_breaker < other.tie_breaker


class MAPFProblem:
    def __init__(self, num_agents: int,
            start_positions: List[Point],
            current_positions: List[Point],
            goal_positions: List[Point],
            grid: Grid,
        ):
        self.num_agents = num_agents
        self.start_positions = start_positions
        self.current_positions = current_positions
        self.goal_positions = goal_positions
        self.grid = grid

    def getNum(self):
        return self.num_agents

    def getStart(self, agent_id: int) -> Node:
        # continuous position -> grid id
        return self.grid.getNode(
            self.start_positions[agent_id] 
        )

    def getCurrent(self, agent_id):
        return self.grid.getNode(
            self.current_positions[agent_id]
        )

    def getGoal(self, agent_id):
        return self.grid.getNode(
            self.goal_positions[agent_id]
        )

    def getConfigStart(self):
        currents = []
        for i in range(self.getNum()):
            currents.append(self.getCurrent(i))
        return currents

class PIBT:
    def __init__(self, problem: MAPFProblem):
        self.problem = problem
        self.occupied_now = [None] * self.problem.grid.num_nodes
        self.occupied_next = [None] * self.problem.grid.num_nodes
        self.solved = False
        self.max_timestep = 30  # Maximum number of timesteps
        self.start_time = time.time()
        self.solution = []
    
    def getElapsedTime(self):
        return f"{time.time() - self.start_time:.2f} seconds"
    
    def getSolution(self):
        paths = [[] for _ in range(self.problem.getNum())]
        for nodes in self.solution:
            for i, n in enumerate(nodes):
                paths[i].append(self.problem.grid.getPoint(n))
        return paths

    def run(self, max_step):
        self.max_timestep = max_step
        undecided = []; decided = []
        for i in range(self.problem.getNum()):
            start = self.problem.getStart(i)
            current = self.problem.getCurrent(i)
            goal = self.problem.getGoal(i)
            d = self.problem.grid.pathDist(start, goal)
            agent = Agent(i, current, goal, 0, d, float(i) / float(self.problem.getNum()))
            heapq.heappush(undecided, agent)
            self.occupied_now[current.id] = agent
        self.solution = [self.problem.getConfigStart()]

        timestep = 0
        while True:
            #print(" ", "elapsed:", self.getElapsedTime(), ", timestep:", timestep)

            while undecided:
                agent = heapq.heappop(undecided)
                if agent.v_next is None:
                    self.funcPIBT(agent)
                decided.append(agent)
            
            check_goal_cond = True
            config = [None] * self.problem.getNum()
            for a in decided:
                if self.occupied_now[a.v_now.id] == a:
                    self.occupied_now[a.v_now.id] = None
                self.occupied_next[a.v_next.id] = None
                # set next location
                config[a.id] = a.v_next
                self.occupied_now[a.v_next.id] = a
                # check goal condition
                check_goal_cond &= (a.v_next == a.g)
                # update priority
                a.elapsed = 0 if (a.v_next == a.g) else a.elapsed + 1
                # reset params
                a.v_now = a.v_next
                a.v_next = None
                # push to priority queue
                heapq.heappush(undecided, a)
            decided.clear()
            # update plan
            self.solution.append(config)

            timestep += 1
            if check_goal_cond:
                self.solved = True
                break
            if timestep >= self.max_timestep:
                break

    def funcPIBT(self, ai: Agent) -> bool:
        v = self.planOneStep(ai)
        while v is not None:
            aj = self.occupied_now[v.id]
            if aj is not None and aj != ai and aj.v_next is None:
                if self.funcPIBT(aj) == False:
                    v = self.planOneStep(ai)
                    continue
            return True
        self.occupied_next[ai.v_now.id] = ai
        ai.v_next = ai.v_now
        return False

    def planOneStep(self, a: Agent) -> Node:
        v = self.chooseNode(a)
        if v is not None:
            self.occupied_next[v.id] = a
            a.v_next = v
        return v

    def chooseNode(self, a: Agent) -> Node:
        C = self.problem.grid.getNeighbors(a.v_now)
        C.append(a.v_now)
        random.shuffle(C)
        v = None
        for u in C:
            if self.occupied_next[u.id] is not None:
                continue
            other = self.occupied_now[u.id]
            if other is not None and other.v_next is not None:
                if other.v_next.id == a.v_now.id:
                    continue
            if u == a.g:
                return u
            if v is None:
                v = u
            else:
                c_v = self.problem.grid.pathDist(a.g, v)
                c_u = self.problem.grid.pathDist(a.g, u)
                if c_u < c_v or (c_u == c_v and self.occupied_now[v.id] is not None and self.occupied_now[u.id] is None):
                    v = u
        return v
