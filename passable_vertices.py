"""
==================
Find passable points from shape vertices
==================

Find the points which the character can pass through 
from the shape vertices

"""

"""
Core functionnality
"""

from shapely.geometry import Polygon

shape_vertices = [
    (0, 0),
    (0, 4),
    (4, 4),
    (4, 0)
]
player_radius = 0.5

polygon = Polygon(shape_vertices)
# Scale the polygon by player_radius*sqrt(2) so that it's away enough to avoid collision
scaled_polygon = polygon.buffer(player_radius*1.4143, join_style=2)

passable_points=[point[0] for point in zip(scaled_polygon.exterior.coords)]

"""
Debug plotting
"""

# # Plot the polygon
# import matplotlib.pyplot as plt

# # Plot the points zip(buffered_polygon.exterior.coords)
# for point in zip(buffered_polygon.exterior.coords):
#     plt.scatter(*point[0], color="red")

# for x,y in shape_vertices:
#     plt.scatter(x,y, color="blue")

# plt.show()