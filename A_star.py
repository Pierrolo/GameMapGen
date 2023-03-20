# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 20:20:59 2023

@author: woill
"""
import heapq

# define the symbols for different types of cells in the grid
EMPTY = 0
OBSTACLE = 1


def heuristic(a, b):
    """Returns the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    """Returns the shortest path from start to goal in a grid using A* algorithm."""
    open_set = [(0, start)]  # Initialize the priority queue with the start node.
    came_from = {}  # Initialize the dictionary to keep track of the path.
    g_score = {start: 0}  # Initialize the dictionary to keep track of the cost to reach each node.
    f_score = {start: heuristic(start, goal)}  # Initialize the dictionary to keep track of the total estimated cost from start to goal through each node.
    
    while open_set:
        current = heapq.heappop(open_set)[1]  # Get the node with the lowest estimated cost from the priority queue.

        if current == goal:
            path = [current]  # Reconstruct the path from the goal to the start.
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in [(current[0]-1, current[1]), (current[0]+1, current[1]), (current[0], current[1]-1), (current[0], current[1]+1)]:  # Check the four neighbors of the current node.
            if neighbor[0] < 0 or neighbor[0] >= len(grid) or neighbor[1] < 0 or neighbor[1] >= len(grid[0]):
                continue  # Skip the neighbor if it is outside the grid.
            if grid[neighbor[0]][neighbor[1]] == OBSTACLE:
                continue  # Skip the neighbor if it is blocked.
            tentative_g_score = g_score[current] + 1  # Calculate the tentative cost to reach the neighbor.
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current  # Update the path to the neighbor.
                g_score[neighbor] = tentative_g_score  # Update the cost to reach the neighbor.
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)  # Update the total estimated cost from start to goal through the neighbor.
                heapq.heappush(open_set, (f_score[neighbor], neighbor))  # Add the neighbor to the priority queue.

    return None  # Return None if there is no path from start to goal.






if __name__ == "__main__": 
    import numpy as np
    grid = [
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, OBSTACLE, EMPTY, OBSTACLE, EMPTY],
        [EMPTY, OBSTACLE, EMPTY, OBSTACLE, EMPTY],
        [EMPTY, OBSTACLE, EMPTY, OBSTACLE, EMPTY],
        [EMPTY, OBSTACLE, EMPTY, OBSTACLE, EMPTY]
    ]
    
    start = (4, 0)
    goal = (4, 4)
    
    path = a_star(np.array(grid), start, goal)
    
    print(path)