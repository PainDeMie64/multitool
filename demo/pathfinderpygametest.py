import numpy as np
import numba
from numba.typed import List
import rustworkx as rx
from shapely.geometry import Polygon

class CollisionChecker:
    def __init__(self, obstacles, player_radius):
        self.collision_obstacles = self.create_buffered_obstacles(obstacles, player_radius)
        self.update_obstacles(self.collision_obstacles)
    
    def update_obstacles(self, collision_obstacles):
        self.collision_polygons = List()
        self.collision_aabbs = List()
        self.collision_edge_aabbs = List()
        for poly in collision_obstacles:
            min_x = np.min(poly[:, 0])
            max_x = np.max(poly[:, 0])
            min_y = np.min(poly[:, 1])
            max_y = np.max(poly[:, 1])
            self.collision_polygons.append(poly)
            self.collision_aabbs.append(np.array([min_x, min_y, max_x, max_y], dtype=np.float64))

            edge_aabbs = []
            n = poly.shape[0]
            for j in range(n):
                a = poly[j]
                b = poly[(j+1) % n]
                edge_aabbs.append([
                    min(a[0], b[0]),
                    min(a[1], b[1]),
                    max(a[0], b[0]),
                    max(a[1], b[1])
                ])
            self.collision_edge_aabbs.append(np.array(edge_aabbs, dtype=np.float64))
    
    def line_intersects_with_any_obstacle(self, p1, p2):
        p1_np = np.array(p1, dtype=np.float64)
        p2_np = np.array(p2, dtype=np.float64)
        return intersects_check(p1_np, p2_np, self.collision_polygons, self.collision_aabbs, self.collision_edge_aabbs)
    
    def point_inside_obstacle(self, point):
        point_np = np.array([point[0], point[1]], dtype=np.float64)
        for poly in self.collision_obstacles:
            if is_inside_sm(poly, point_np):
                return True
        return False
    
    def create_buffered_obstacles(self, obstacles, buffer_distance):
        buffered = []
        for obstacle in obstacles:
            poly = Polygon(obstacle)
            buffered_poly = poly.buffer(buffer_distance, quad_segs=5)
            exterior = np.array(buffered_poly.exterior.coords[:-1], dtype=np.float64)
            buffered.append(exterior)
        return buffered
    
@numba.njit(fastmath=True)
def intersects_check(p1, p2, collision_polygons, collision_aabbs, edge_aabbs):
    min_x_seg = min(p1[0], p2[0])
    max_x_seg = max(p1[0], p2[0])
    min_y_seg = min(p1[1], p2[1])
    max_y_seg = max(p1[1], p2[1])
    points = np.stack((p1, p2))
    for i, poly in enumerate(collision_polygons):
        aabb = collision_aabbs[i]
        poly_min_x, poly_min_y, poly_max_x, poly_max_y = aabb
        if (poly_min_x > max_x_seg) or (poly_max_x < min_x_seg) or (poly_min_y > max_y_seg) or (poly_max_y < min_y_seg):
            continue
        for point in points:
            if is_inside_sm(poly, point):
                return True
        n = poly.shape[0]
        for j in range(n):
            min_x_edge, min_y_edge, max_x_edge, max_y_edge = edge_aabbs[i][j]
            
            if (min_x_edge > max_x_seg) or (max_x_edge < min_x_seg) or (min_y_edge > max_y_seg) or (max_y_edge < min_y_seg):
                continue
            
            a = poly[j]
            b = poly[(j+1) % n]
            if segments_intersect(p1, p2, a, b):
                return True
    
    return False

@numba.njit("boolean(float64[:, :], float64[:])")
def is_inside_sm(polygon, point):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
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

@numba.njit("float64(float64[:], float64[:], float64[:])")
def ccw(A, B, C):
    return (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])

@numba.njit("boolean(float64[:], float64[:], float64[:])")
def on_segment(A, B, C):
    return (min(A[0], C[0]) <= B[0] <= max(A[0], C[0])) and (min(A[1], C[1]) <= B[1] <= max(A[1], C[1]))

@numba.njit("boolean(float64[:], float64[:], float64[:], float64[:])")
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

@numba.njit("float64(UniTuple(float64, 2), UniTuple(float64, 2))", fastmath=True)
def get_weight(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

class Pathfinder:
    def __init__(self, obstacles, player_radius, collision_offset=-1, path_offset=2):
        self.obstacles = [np.array(obs, dtype=np.float64) for obs in obstacles]
        self.collision_checker = CollisionChecker(self.obstacles, player_radius+collision_offset)
        self.path_obstacles = self.collision_checker.create_buffered_obstacles(self.obstacles, player_radius + path_offset)
        self.player_radius = player_radius
        self.passable_points = self._generate_passable_points()
        self.concatenated_passable_points = self._generate_concatenated_passable_points()
        self.graph = rx.PyGraph()
        self._build_base_graph()

        self.width_interval = 0
        self.height_interval = 0
        self.preprocessed_points = []
        self.preprocessed_paths = dict()
        self.last_path = []
    
    def _generate_passable_points(self):
        return [obstacle for obstacle in self.path_obstacles]
    
    def _generate_concatenated_passable_points(self):
        return [tuple(point) for obstacle in self.passable_points for point in obstacle]

    def _build_base_graph(self):
        self.graph.extend_from_weighted_edge_list([])
        self.graph.add_nodes_from(self.concatenated_passable_points)
        n = len(self.concatenated_passable_points)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.concatenated_passable_points[i]
                p2 = self.concatenated_passable_points[j]
                if not self.collision_checker.line_intersects_with_any_obstacle(p1, p2):
                    weight = get_weight(p1, p2) * self._speed_multiplier(p1, p2)
                    self.graph.add_edge(i, j, weight)
    
    def _speed_multiplier(self, p1, p2):
        return 1.0
    
    def find_path(self, start_pos, end_pos):
        temp_graph = self.graph.copy()
        points = self.concatenated_passable_points
        n_original = len(points)
        
        start_node_idx = temp_graph.add_node(start_pos)
        end_node_idx = temp_graph.add_node(end_pos)
        
        if not self.collision_checker.line_intersects_with_any_obstacle(start_pos, end_pos):
            weight = get_weight(start_pos, end_pos) * self._speed_multiplier(start_pos, end_pos)
            temp_graph.add_edge(start_node_idx, end_node_idx, weight)
        else:
            for i in range(n_original):
                point = points[i]
                if not self.collision_checker.line_intersects_with_any_obstacle(start_pos, point):
                    weight = get_weight(start_pos, point) * self._speed_multiplier(start_pos, point)
                    temp_graph.add_edge(start_node_idx, i, weight)
                if not self.collision_checker.line_intersects_with_any_obstacle(end_pos, point):
                    weight = get_weight(end_pos, point) * self._speed_multiplier(end_pos, point)
                    temp_graph.add_edge(end_node_idx, i, weight)
        
        try:
            path_indices = rx.astar_shortest_path(
                temp_graph,
                start_node_idx,
                goal_fn=lambda data: data == end_pos,
                edge_cost_fn=lambda e: e,
                estimate_cost_fn=lambda pos: get_weight(pos, end_pos)
            )
            if path_indices is None:
                return None
            path_points = [temp_graph[node_idx] for node_idx in path_indices]
            return path_points
        except rx.NoPathFound:
            return self.last_path
        
    def get_move_direction(self, current_pos, target_pos):
        if(len(self.preprocessed_paths)==0):
            path = self.find_path(current_pos, target_pos)
        else:
            path = self.find_prepocessed_path(current_pos, target_pos)
        if path is None or len(path) < 2:
            return (0, 0)
        self.last_path = path
        next_point = path[1]
        dx = next_point[0] - current_pos[0]
        dy = next_point[1] - current_pos[1]
        return (dx, dy)
    
    # def find_prepocessed_path(self, start_pos, end_pos):
    #     start_x_idx, start_y_idx = int(start_pos[0]/self.width_interval), int(start_pos[1]/self.height_interval)
    #     end_x_idx, end_y_idx = int(end_pos[0]/self.width_interval), int(end_pos[1]/self.height_interval)
        
    #     if start_x_idx+1>=len(self.preprocessed_points) or start_y_idx+1>=len(self.preprocessed_points) or end_x_idx+1>=len(self.preprocessed_points) or end_y_idx+1>=len(self.preprocessed_points):
    #         print("Error: Preprocessed points are not enough")
    #         return self.find_path(start_pos, end_pos)
    #     start_corners=[(start_x_idx*self.width_interval, start_y_idx*self.height_interval),
    #                       ((start_x_idx+1)*self.width_interval, start_y_idx*self.height_interval),
    #                       (start_x_idx*self.width_interval, (start_y_idx+1)*self.height_interval),
    #                       ((start_x_idx+1)*self.width_interval, (start_y_idx+1)*self.height_interval)]
    #     end_corners=[(end_x_idx*self.width_interval, end_y_idx*self.height_interval),
    #                         ((end_x_idx+1)*self.width_interval, end_y_idx*self.height_interval),
    #                         (end_x_idx*self.width_interval, (end_y_idx+1)*self.height_interval),
    #                         ((end_x_idx+1)*self.width_interval, (end_y_idx+1)*self.height_interval)]
    #     for corner in start_corners:
    #         if not self.collision_checker.line_intersects_with_any_obstacle(start_pos, corner):
    #             reachable_start_point=corner
    #             break
    #     else:
    #         return self.find_path(start_pos, end_pos)
    #     for corner in end_corners:
    #         if not self.collision_checker.line_intersects_with_any_obstacle(end_pos, corner):
    #             reachable_end_point=corner
    #             break
    #     else:
    #         return self.find_path(start_pos, end_pos)
    #     if reachable_start_point==reachable_end_point:
    #         return self.find_path(start_pos, end_pos)
    #     try:
    #         closest_path=self.preprocessed_paths[(reachable_start_point, reachable_end_point)]
    #     except KeyError:
    #         closest_path=self.preprocessed_paths[(reachable_end_point, reachable_start_point)][::-1]
    #     return [start_pos]+closest_path[1:-1]+[end_pos]
    

    # def preprocess_paths(self, resolution_x, resolution_y, width, height):
    #     self.width_interval = width / resolution_x
    #     self.height_interval = height / resolution_y
    #     self.preprocessed_points = []
    #     for i in range(resolution_x):
    #         x = i * self.width_interval
    #         for j in range(resolution_y):
    #             y = j * self.height_interval
    #             point = (x, y)
    #             self.preprocessed_points.append(point)
    #     self.preprocessed_paths = {}
    #     prog = 0
    #     total = len(self.preprocessed_points) * (len(self.preprocessed_points) - 1) / 2
    #     for i, point in enumerate(self.preprocessed_points):
    #         if i%10==0:
    #             print(f"Progress: {prog}/{total}")
    #         for j in range(i + 1, len(self.preprocessed_points)):
    #             prog+=1
    #             target_point = self.preprocessed_points[j]
    #             try:
    #                 path = self.find_path(point, target_point)
    #                 if path is not None:
    #                     self.preprocessed_paths[(point, target_point)] = path
    #             except rx.NoPathFound:
    #                 continue


import pygame
import random
import math
import numpy as np
import numba
from numba import boolean, float64
from shapely.geometry import Polygon, Point
import timeit
import time

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Optimized Visibility Graph Test")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

PLAYER_SIZE = 10
PLAYER_SPEED = 3

GOAL_SIZE = 15

pygame.font.init()
FONT = pygame.font.Font(None, 24)

obstacles = [
    np.array([[100, 100], [250, 100], [250, 250], [100, 250]], dtype=np.float64),
    np.array([[500, 400], [650, 400], [575, 300]], dtype=np.float64),
    np.array([[400, 100], [450, 50], [550, 50], [600, 100],
              [550, 150], [450, 150]], dtype=np.float64)
]

class Player:
    def __init__(self):
        self.x = 653
        self.y = 319
        self.color = BLUE

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), PLAYER_SIZE)

    def respawn(self):
        self.x = random.randint(50, WIDTH-50)
        self.y = random.randint(50, HEIGHT-50)

class Goal:
    def __init__(self):
        self.x = 105
        self.y = 542
        self.color = GREEN

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), GOAL_SIZE)

    def respawn(self):
        self.x = random.randint(50, WIDTH-50)
        self.y = random.randint(50, HEIGHT-50)

def main():
    running = True
    clock = pygame.time.Clock()

    player = Player()
    goal = Goal()
    iteration=0
    
    pathfinder = Pathfinder(obstacles, player_radius=PLAYER_SIZE) 
    # pathfinder.preprocess_paths(20, 20, WIDTH, HEIGHT)
    collision_checker = CollisionChecker(obstacles, player_radius=PLAYER_SIZE+2)
    
    while running:
        iteration+=1
        
        # if iteration%200==0:
        #     print(timeit.timeit(lambda: pathfinder.get_move_direction((player.x,player.y), (goal.x,goal.y)), number=1000)/1000)
            
        screen.fill(WHITE)

        # Draw obstacles
        for poly in obstacles:
            pygame.draw.polygon(screen, RED, poly, 0)
            #pygame.draw.polygon(screen, BLACK, poly, 2)

        # Draw player and goal
        player.draw()
        goal.draw()


        # Handle input
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0

        if keys[pygame.K_LEFT]:
            dx -= PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            dx += PLAYER_SPEED
        if keys[pygame.K_UP]:
            dy -= PLAYER_SPEED
        if keys[pygame.K_DOWN]:
            dy += PLAYER_SPEED

        dx, dy = pathfinder.get_move_direction((player.x,player.y), (goal.x,goal.y))

        dx, dy = np.array([dx, dy]) / np.linalg.norm([dx, dy]) * PLAYER_SPEED

        new_x = player.x + dx
        new_y = player.y + dy
        if 0 <= new_x <= WIDTH and 0 <= new_y <= HEIGHT:
            player.x = new_x
            player.y = new_y

        # Check goal reached
        distance = math.hypot(player.x - goal.x, player.y - goal.y)
        if distance < PLAYER_SIZE + GOAL_SIZE:
            player.respawn()
            while collision_checker.point_inside_obstacle((player.x,player.y)):
                player.respawn()
            goal.respawn()
            while collision_checker.point_inside_obstacle((goal.x,goal.y)):
                goal.respawn()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()