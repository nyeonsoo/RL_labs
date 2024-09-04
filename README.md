# rl_labs
Lab materials for the Reinforcement Learning class

## Install

```
git clone https://github.com/inmo-jang/rl_labs.git
cd rl_labs
pip install -r requirements.txt
```


## Simple Maze Grid

![image](https://github.com/user-attachments/assets/52d096ee-f5bf-400b-a2d3-deae3fc2b628)


**Manual Play**

```
python gyms/simple_maze_grid.py
```

- Keyboard Control
  - `q` or `ESC`: quit
  - `r`: reset


**Import**
```
env = SimpleMazeGrid(n=5, k=3, m=2, render_option=True, random_seed=42)

# n = Length of one side of the grid
# k = Starting index from which the goal can be generated, extending to the end of the grid
# m = Number of obstacles
```


- Actions: `Discrete(2)`
- Observation Space: `Discrete(n * n)`
- Reward:
  - `10` when arriving the goal
  - `-10` when arriving a pit
  - `-1` otherwise
