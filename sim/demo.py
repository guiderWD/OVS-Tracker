import os
import copy
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from src.bfs import bfs
from src.utils import Point, State
from src.pibt import Grid, MAPFProblem, PIBT 
from src.costmap import CostMap
from src.trajectory import Point, PathSearcher, TrajectoryOptimizer
from src.orientation import VelocityController
from src.waypoint import WayPointOptimizer
from typing import List

def is_acute(a: Point, b: Point, c: Point):
    ba = Point(a.x - b.x, a.y - b.y)
    bc = Point(c.x - b.x, c.y - b.y)
    dot_product = ba.x * bc.x + ba.y * bc.y
    return dot_product >= 0


def plot_points(path: List[Point]):
    xs = []; ys = []
    for point in path:
        xs.append(point.x)
        ys.append(point.y)
    colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))
    # plt.plot(xs, ys, "--", markersize=3, label="trajectory")
    plt.scatter(xs, ys, c=colors, label="trajectory")
    for i in range(len(xs)):
        plt.text(xs[i], ys[i], f"{i}", c="r")


def plot_states(states: List[State]):
    for i, state in enumerate(states):
        dx = np.cos(state.theta) * 30
        dy = np.sin(state.theta) * 30
        if i% 20 == 0:
            plt.arrow(state.x, state.y, dx, dy,
                width = 5, linewidth = 0, head_length = 15,
                length_includes_head = True, facecolor = "C3",
                label="states" if i == 0 else None)


def plot_agents(states: List[State]):
    for i, state in enumerate(states):
        dx = np.cos(state.theta) * 30
        dy = np.sin(state.theta) * 30
        plt.arrow(state.x, state.y, dx, dy,
            width = 20, linewidth = 0, head_length = 20,
            length_includes_head = True, facecolor = f"C{i}")


def main():
    map_array = io.imread("map/711_casia.pgm")
    # 垂直镜像翻转
    map_array = np.flipud(map_array)
    map_array = map_array.astype(np.uint8)
    expanded_map_array, esdf = CostMap(map_array).map2esdf()
    # load topological graphs
    grid = Grid()
    # define initial state of the agents
    num_agents = 4
    start_positions = [
        Point(250, 50), Point(1050, 60), Point(815, 175) ,Point(300, 200)
    ]
    goal_positions = [
        Point(1440, 750), Point(700, 50), Point(250, 50), Point(815, 175)
    ]
    start_thetas = [
        0, np.pi , -np.pi / 3 ,0
    ]

    num_agents = 10
    start_positions = [
        Point(250, 900), Point(250, 700), Point(250, 600) ,Point(500, 400),Point(500, 600),Point(400, 200),Point(1050, 60), Point(815, 175),Point(50, 200),Point(700, 400)
    ]
    # goal_positions = [
    #     Point(1000, 500), Point(1000, 500), Point(1000, 500) ,Point(1000, 500),Point(1000, 500),Point(1000, 500),Point(1000, 500),Point(1000, 500),Point(1000, 500),Point(1000, 500)
    # ]
    goal_positions = [
        Point(1400, 700), Point(1600, 400), Point(1400, 600) ,Point(1600, 800),Point(1400, 600),Point(900, 400),Point(1100, 600),Point(900, 500),Point(1100, 900),Point(1000, 400)
    ]
    start_thetas = [
        0, np.pi , -np.pi / 3 ,0,0,0,0,0,0,0
    ]
    
    assert len(start_positions) == num_agents
    assert len(goal_positions) == num_agents
    assert len(start_thetas) == num_agents
    
    states =[
        State(sp.x, sp.y, st, dt=0.02)
        for sp, st in zip(start_positions, start_thetas)
    ]
    current_positions = copy.deepcopy(start_positions)

    count = 0
    step = 0; max_step = 50
    while step < max_step:
        problem = MAPFProblem(
            num_agents,
            start_positions,
            current_positions,
            goal_positions,
            grid,
        )
        pibt = PIBT(problem)
        pibt.run(max_step=50)
        step += 1
        print(step)
        paths = pibt.getSolution()
        # from topological graph to 2-d positions on high-resolution maps
        paths = [
            [Point(*bfs(255 - expanded_map_array, int(p.x), int(p.y), 0))
            for p in path] for path in paths]
        # use the quantized current position as the first waypoint
        # notion: current position and the quantized one can be far away, the agents
        # must arrive at the quantized current position first, and then change
        # the waypoint to the second point on the path
        pre_waypoints = [paths[i][0] for i in range(num_agents)]
        waypoints = [paths[i][1] for i in range(num_agents)]
        next_waypoints = [paths[i][2] for i in range(num_agents)]
        # indicating whether the agents arrive at the first waypoint (i.e., quantized
        # current positions)
        not_arrive_at_first_waypoint = np.array([True for _ in range(num_agents)])
        # indicating whether the agents arrive at the second waypoint
        not_arrive_at_second_waypoint = np.array([True for _ in range(num_agents)])

        #not_arrive_at_second_waypoint[0] = False
        # False: use first waypoint as subgoal
        # True:  use second waypoint as subgoal
        second_waypoint_as_goal = False
        #print("not_arrive_at_second_waypoint.any")
        # loop when one of the agents has not arrived at the second waypoint
        update_system_wd=False

        while not_arrive_at_second_waypoint.any() and update_system_wd == False:
            print("not_arrive_at_second_waypoint")
            print(not_arrive_at_second_waypoint)
            print("  ")
            print("not_arrive_at_first_waypoint")
            print(not_arrive_at_first_waypoint)
            print("  ")
            #print()
            subgoals = waypoints  # 使用初始的导航点，而不经过优化器
            plot_flag = False
            pthalls = []
            #print("waypoints")
            #print(waypoints)
            for i in range(num_agents):
                state, waypoint, subgoal = states[i], waypoints[i], subgoals[i]
                g = Point(goal_positions[i].x, goal_positions[i].y)
                current_pt = Point(int(state.x), int(state.y))
                # if the agent arrive at the waypoint
                #next_waypoints=None
                # if i == 0 :      
                #     print(np.sqrt((next_waypoints[i].x - pre_waypoints[i].x) ** 2 + (next_waypoints[i].y - pre_waypoints[i].y) ** 2) >= 150 and  np.sqrt((next_waypoints[i].x - current_pt.x) ** 2 + (next_waypoints[i].y - current_pt.y) ** 2) < 100)
                if np.sqrt((next_waypoints[i].x - pre_waypoints[i].x) ** 2 + (next_waypoints[i].y - pre_waypoints[i].y) ** 2) >= 150 and  np.sqrt((next_waypoints[i].x - current_pt.x) ** 2 + (next_waypoints[i].y - current_pt.y) ** 2) < 100:
                    not_arrive_at_second_waypoint[i] = False
                    not_arrive_at_first_waypoint[i] = False

                # 计算 False 的数量
                num_false = np.sum(not_arrive_at_first_waypoint == False)

                # 检查是否超过一半
                if num_false > num_agents / 2:
                    update_system_wd=True
                    print("update")
                    #update太快会导致车辆反复纠结，太慢会导致其它到达车辆等待时间过长

                if np.sqrt((current_pt.x - waypoint.x) ** 2 + (current_pt.y - waypoint.y) ** 2) < 20.0 :
                    
                    # print("ininininnini")
                    # print(not_arrive_at_first_waypoint[i])
                    # print("ininininnini")
                    if not_arrive_at_first_waypoint[i]: #这里为true时进入
                        not_arrive_at_first_waypoint[i] = False
                        #print("ininininninieeeerrrr")
                    if not not_arrive_at_first_waypoint.any():
                        if second_waypoint_as_goal: 
                            not_arrive_at_second_waypoint[i] = False
                        else:
                            pre_waypoints = [paths[i][0] for i in range(num_agents)]
                            waypoints = [paths[i][1] for i in range(num_agents)]
                            next_waypoints=[paths[i][2] for i in range(num_agents)]
                            second_waypoint_as_goal = True 
                    #print(not_arrive_at_second_waypoint)
                    continue

                self_subgoal_occupied = False
                for j in range(num_agents):
                    if j != i:  # 不检查自己
                        other_current_pt = Point(int(states[j].x), int(states[j].y))
                        if np.sqrt((subgoal.x - other_current_pt.x) ** 2 + (subgoal.y - other_current_pt.y) ** 2) < 60.0: 
                            self_subgoal_occupied = True
                            break
                occupied_others_subgoal = False
                for j in range(num_agents):
                    if j != i:  # 不检查自己
                        other_current_subgoal = Point(int(subgoals[j].x), int(subgoals[j].y))
                        if np.sqrt((state.x - other_current_subgoal.x) ** 2 + (state.y - other_current_subgoal.y) ** 2) < 5.0:
                            occupied_others_subgoal = True
                            break
                if occupied_others_subgoal == True and self_subgoal_occupied == True:
                    print("卡死啦")

                next_point_not_occpy=False
                num_occpy_nx=0
                for j in range(num_agents):
                    if j != i:  # 不检查自己
                        other_current_pt = Point(int(states[j].x), int(states[j].y))
                        if np.sqrt((next_waypoints[i].x - other_current_pt.x) ** 2 + (next_waypoints[i].y - other_current_pt.y) ** 2) < 100 or  np.sqrt((next_waypoints[i].x - pre_waypoints[i].x) ** 2 + (next_waypoints[i].y - pre_waypoints[i].y) ** 2) < 150:
                            next_point_not_occpy = False
                            break
                        else:
                            num_occpy_nx = num_occpy_nx + 1
                        if num_occpy_nx == num_agents - 1:
                            next_point_not_occpy = True
                # if self_subgoal_occupied :
                #     pre_waypoint = pre_waypoints[i]
                #     quarter_point = Point(
                        
                #         int(pre_waypoint.x + (subgoal.x - pre_waypoint.x) / 2),
                #         int(pre_waypoint.y + (subgoal.y - pre_waypoint.y) / 2)
                #     )
                #     if  np.sqrt((quarter_point.x - current_pt.x) ** 2 + (quarter_point.y - current_pt.y) ** 2) <10.0:
                #         pth = [current_pt]  # 或者直接返回当前点作为路径
                #     else:
                #         pth = PathSearcher().plan(expanded_map_array, current_pt, quarter_point)
                #         pth = TrajectoryOptimizer(esdf).plan(expanded_map_array, pth)
                if self_subgoal_occupied:
                    pre_waypoint = pre_waypoints[i]
                    optimize_point = Point(
                        int(pre_waypoint.x + (subgoal.x - pre_waypoint.x) / 2),
                        int(pre_waypoint.y + (subgoal.y - pre_waypoint.y) / 2)
                    )
                    # 计算路径
                    if np.sqrt((optimize_point.x - current_pt.x) ** 2 + (optimize_point.y - current_pt.y) ** 2) < 10.0:
                        pth = [current_pt]  # 或者直接返回当前点作为路径
                    else:
                        pth = PathSearcher().plan(expanded_map_array, current_pt, optimize_point)
                        pth = TrajectoryOptimizer(esdf).plan(expanded_map_array, pth)
                elif next_point_not_occpy :
                    next_waypoint=next_waypoints[i]
                    if np.sqrt((next_waypoint.x - current_pt.x) ** 2 + (next_waypoint.y - current_pt.y) ** 2) <10.0:
                        
                        pth = [current_pt]  # 或者直接返回当前点作为路径
                    else:
                        #pth = [current_pt,subgoal,next_waypoint]
                        pth = PathSearcher().plan(expanded_map_array, current_pt, next_waypoint)
                        pth = TrajectoryOptimizer(esdf).plan(expanded_map_array, pth)
                    print("gogoggo")
                else:
                    if np.sqrt((subgoal.x - current_pt.x) ** 2 + (subgoal.y - current_pt.y) ** 2) <10.0:
                        pth = [current_pt]  # 或者直接返回当前点作为路径
                    else:
                        pth = PathSearcher().plan(expanded_map_array, current_pt, subgoal)
                        pth = TrajectoryOptimizer(esdf).plan(expanded_map_array, pth)   
                # 保存当前代理的路径到 pthalls
                pthalls.append(pth)

                # not_arrive_at_second_waypoint[0] = False
                # not_arrive_at_first_waypoint[0] = False
                # if i != 0:
                velocity_controller = VelocityController(
                    pth, copy.deepcopy(g), Point(1000, 500)
                )
                # update the states of the agents with velocity controller
                for _ in range(10):
                    velocity, path_point = velocity_controller.control(
                        state.x, state.y, state.theta
                    )
                    state.update(velocity)
                plot_flag = True
            
            current_positions = [
                Point(states[i].x, states[i].y)
                for i in range(num_agents)
            ]

            if plot_flag:
                count += 1
                plt.figure(figsize=(12.0, 6.0))
                plt.imshow(expanded_map_array, cmap="gray")
                for i in range(num_agents):
                    # 获取当前代理的路径
                    agent_path = pthalls[i] if i < len(pthalls) else []  #pthalls pthalls
                    path_x = [pt.x for pt in agent_path]
                    path_y = [pt.y for pt in agent_path]
                    plt.plot(path_x, path_y, label=f"Path: Agent {i}", linestyle='--')  # 绘制路径


                    agent_path = paths[i] if i < len(paths) else []  #pthalls pthalls
                    path_x = [pt.x for pt in agent_path]
                    path_y = [pt.y for pt in agent_path]
                    plt.plot(path_x, path_y, label=f"Pathsuper: Agent {i}", linestyle='--')  # 绘制路径
                    
                    subgoal = subgoals[i]  # 获取当前代理的 subgoal
                    plt.scatter(subgoal.x, subgoal.y, marker='o', s=30, label=f"Subgoal: Agent {i}")
                for i in range(num_agents):
                    plt.plot(
                        goal_positions[i].x, goal_positions[i].y,
                        marker='x', markersize=20, label=f"Goal: Agent {i}"
                    )
                plot_agents(states)
                plt.legend()
                plt.savefig(f"photo/10agent/agents_count_{count:03d}.png")
                plt.close()
                print("showwwww111111111111111111")
                plot_flag = False

if __name__ == "__main__":
    main()

    
                ###在这里加入waypoints占用机制，除开本循环中自身位置，其他agents如果离得[paths[i][1] for i in range(num_agents)]较近，则不要更新waypoints，以此来防止与其它agents撞击
                #print("type(pth)")
                #print(type(pth))
                #print("type(pth)")