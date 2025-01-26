# Pathfinder: Intelligent Obstacle Navigation Library  
*Optimal pathfinding with collision avoidance for dynamic environments*  

## 🔍 Overview  
A high-performance Python implementation using visibility graphs and computational geometry to find optimal obstacle-avoiding paths. Features Minkowski sum-based collision buffers for safe navigation around obstacles.

## ✨ Key Features  
- **Visibility Graph Optimization**: Efficiently computes shortest paths through obstacle vertices  
- **Collision Buffer System**: Uses Minkowski sums to maintain safe distances from obstacles  
- **Dynamic Recalculation**: Real-time path updates for moving targets  
- **Polygonal Support**: Works with arbitrary obstacle shapes  
- **Precision Navigation**: Guaranteed shortest continuous path

## 🚀 Quick Start  

### Basic Usage  
```python  
from pathfinder import Pathfinder  

# Initialize with game environment  
OBSTACLES = [...]  # List of polygonal obstacles  
PLAYER_RADIUS = 1.2  

pathfinder = Pathfinder(obstacles=OBSTACLES, player_radius=PLAYER_RADIUS)  

# Calculate movement vector  
current_pos = (x, y)  
target_pos = (goal_x, goal_y)  

dx, dy = pathfinder.get_move_direction(current_pos, target_pos)  
```

## 📅 Roadmap
- [ ] Dynamic obstacle support
- [ ] Multi-agent collision avoidance  

---