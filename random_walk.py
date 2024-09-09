"""
=====================
Random Walk Algorithm
=====================

Implementing a simple random walk algorithm for a game bot to 
simulate unpredictable movement.
"""

"""
------------------
Core functionality
------------------
"""
import random

"""Input"""
positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # Representing positions on a grid or map
steps = 5

"""Calculation & Output"""
path=[random.choice(positions) for _ in range(steps)]