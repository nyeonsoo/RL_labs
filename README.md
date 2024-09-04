# rl_labs
Lab materials for the Reinforcement Learning class




## Simple Maze Grid

**Manual Play**

```
python gyms/simple_maze_grid.py
```


**Import**
```
env = SimpleMazeGrid(n=5, k=3, m=2, render_option=True, random_seed=42)

# n = Length of one side of the grid
# k = Starting index from which the goal can be generated, extending to the end of the grid
# m = Number of obstacles
```


- Actions: `Discrete(2)`
- Observation Space: `Discrete(n * n)`