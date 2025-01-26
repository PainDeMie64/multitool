import math
import numpy as np
import numba
import networkx as nx
from shapely.geometry import Polygon

@numba.njit("boolean(float64[:, :], float64[:])")
def is_inside_sm(polygon, point):
    length = len(polygon)
    inside = False
    x, y = point
    for i in range(length):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % length]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

@numba.njit("boolean[:](float64[:, :], float64[:, :])")
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon, points[i])
    return D

@numba.njit
def ccw(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

@numba.njit
def on_segment(A, B, C):
    return (
        (min(A[0], C[0]) <= B[0] <= max(A[0], C[0])) and
        (min(A[1], C[1]) <= B[1] <= max(A[1], C[1]))
    )

@numba.njit
def segments_intersect(A, B, C, D):
    ccw1 = ccw(A, B, C)
    ccw2 = ccw(A, B, D)
    ccw3 = ccw(C, D, A)
    ccw4 = ccw(C, D, B)

    if (ccw1 * ccw2 < 0) and (ccw3 * ccw4 < 0):
        return True

    if ccw1 == 0 and on_segment(A, C, B):
        return True
    if ccw2 == 0 and on_segment(A, D, B):
        return True
    if ccw3 == 0 and on_segment(C, A, D):
        return True
    if ccw4 == 0 and on_segment(C, B, D):
        return True

    return False

@numba.njit
def get_weight(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def create_buffered_obstacles(obstacles, buffer_distance):
    buffered = []
    for obstacle in obstacles:
        poly = Polygon(obstacle)
        buffered_poly = poly.buffer(buffer_distance, quad_segs=5)
        exterior = np.array(buffered_poly.exterior.coords[:-1], dtype=np.float64)
        buffered.append(exterior)
    return buffered

class Pathfinder:
    def __init__(self, obstacles, player_radius):
        self.obstacles = [np.array(obs, dtype=np.float64) for obs in obstacles]
        self.path_obstacles = create_buffered_obstacles(self.obstacles, player_radius+2)
        self.collision_obstacles = create_buffered_obstacles(self.obstacles, player_radius+1)
        self.player_radius = player_radius
        self.passable_points = self._generate_passable_points()
        self.concatenated_passable_points = self._generate_concatenated_passable_points()
        self.graph = nx.Graph()
        self._build_base_graph()
    
    def _generate_passable_points(self):
        return [obstacle.tolist() for obstacle in self.path_obstacles]
    
    def _generate_concatenated_passable_points(self):
        return  [tuple(point) for obstacle in self.passable_points for point in obstacle]

    def _build_base_graph(self):
        self.graph.add_nodes_from([str(i) for i in range(len(self.concatenated_passable_points))])
        for i,p in enumerate(self.concatenated_passable_points):
            for j in range(i+1, len(self.concatenated_passable_points)):
                if not self._intersects_with_any_obstacle(self.concatenated_passable_points[i], self.concatenated_passable_points[j]):
                    weight = get_weight(self.concatenated_passable_points[i], self.concatenated_passable_points[j]) * self._speed_multiplier(self.concatenated_passable_points[i], self.concatenated_passable_points[j])
                    self.graph.add_edge(str(i), str(j), weight=weight)
    
    def _intersects_with_any_obstacle(self, p1, p2):
        points = np.array([p1, p2], dtype=np.float64)
        for poly in self.collision_obstacles:
            poly_np = np.array(poly, dtype=np.float64)
            if is_inside_sm(poly_np, points[0]) or is_inside_sm(poly_np, points[1]):
                return True
        for poly in self.collision_obstacles:
            poly_np = np.array(poly, dtype=np.float64)
            n = len(poly_np)
            for i in range(n):
                a = poly_np[i]
                b = poly_np[(i + 1) % n]
                if segments_intersect(np.array(p1), np.array(p2), a, b):
                    return True
        return False
    
    def _speed_multiplier(self, p1, p2):
        return 1.0
    
    def find_path(self, start_pos, end_pos):
        temp_graph = self.graph.copy()
        points = self.concatenated_passable_points
        start_node = str(len(points))
        end_node = str(len(points)+1)
        
        temp_graph.add_node(start_node)
        temp_graph.add_node(end_node)
        for i, point in enumerate(points):
            if not self._intersects_with_any_obstacle(start_pos, point):
                temp_graph.add_edge(start_node, str(i), weight=get_weight(start_pos, point)*self._speed_multiplier(start_pos, point))
            if not self._intersects_with_any_obstacle(end_pos, point):
                temp_graph.add_edge(end_node, str(i), weight=get_weight(end_pos, point)*self._speed_multiplier(end_pos, point))
        
        if not self._intersects_with_any_obstacle(start_pos, end_pos):
            temp_graph.add_edge(start_node, end_node, weight=get_weight(start_pos, end_pos)*self._speed_multiplier(start_pos, end_pos))
        try:
            path = nx.shortest_path(temp_graph, start_node, end_node, weight='weight')
            path_points = []
            for node in path:
                if node == start_node:
                    path_points.append(start_pos)
                elif node == end_node:
                    path_points.append(end_pos)
                else:
                    path_points.append(self.concatenated_passable_points[int(node)])
            return path_points
        except nx.NetworkXNoPath:
            return None
    
    def get_move_direction(self, current_pos, target_pos):
        path = self.find_path(current_pos, target_pos)
        if not path or len(path) < 2:
            print("no path")
            return (0.0, 0.0)
        next_point = path[1]
        dx = next_point[0] - current_pos[0]
        dy = next_point[1] - current_pos[1]
        return (dx, dy)
    