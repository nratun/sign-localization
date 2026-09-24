#!/usr/bin/env python

"""
map.py: This file handles floor plan graph creation/visualization and robot movement.

The map uses a predefined graph from the floor-plan YAML file.
Recognized room signs are used to determine the location to move to, and the robot is moved
along the graph to reach it.
"""
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from floor_data import FLOORS_DIR

MOVE_SPEED = 200.0 # Movement speed of robot on map

@dataclass
class Vertex:
    x: float
    y: float
    label: str | None = None

def build_graph(floor_data: dict) -> tuple[dict[int, Vertex], list[tuple[int, int]]]:
    '''
    Takes in a dictionary containing data about a particular building floor.
    Returns all the vertices and edges of the rooms on that floor

    Params:
        floor_data (dict): The information of a particular building floor

    Returns:
        vertices (dict[int, Vertex]]), edges (list[tuple]): The vertices & edges of the graph
    '''
    vertices = {}

    for vertex_id, vertex in enumerate(floor_data["vertices"]):
        x = vertex[0]
        y = vertex[1]
        label = vertex[3]

        if label:
            label = str(label).upper()

        vertices[vertex_id] = Vertex(x=x, y=y, label=label)
    edges = []

    for edge in floor_data["lanes"]:
        start = edge[0]
        end = edge[1]
        edges.append((start, end))
    return vertices, edges

class Map:
    def __init__(
        self,
        floor_data: dict,
        vertices: dict[int, Vertex],
        edges: list[tuple[int, int]],
        rooms: dict[str, int],
        move_speed: float = MOVE_SPEED
    ):
        '''
        Initialize the Map, graph, and robot location

        Params:
            floor_data (dict): The information of a particular building floor
            vertices (dict[int, Vertex]]): The vertices of the graph representing locations
            edges (list[tuple[int, int]]): The edges of the graph connecting the vertices
            rooms (dict[str, int]): The numbered/named rooms on a particular floor & their vertex IDs
            move speed (float): The speed of the robot as it navigates on the map
        '''
        self.vertices = vertices
        self.edges = edges
        self.rooms = rooms
        self.move_speed = move_speed

        # Load floor map as background
        drawing_name = floor_data["drawing"]["filename"]
        drawing_path = FLOORS_DIR / drawing_name
        self.background = cv2.imread(str(drawing_path))

        if self.background is None:
            raise FileNotFoundError(f"Could not load floor plan: {drawing_path}")

        # Render map once
        self.map = self._render_bg()

        # Build adjacency list for graph traversal
        self.adjacency: dict[int, list[int]] = {vertex_id: [] for vertex_id in self.vertices}

        # Add each edge in both directions
        for start_id, end_id in self.edges:
            self.adjacency[start_id].append(end_id)
            self.adjacency[end_id].append(start_id)
        # ---------------------- Robot Location ----------------------
        self.curr_vtx: int | None = None # Last confirmed vertex
        self.curr_room: str | None = None # Last confirmed room
        self.curr_point: tuple[float, float] | None = None # Current robot position
        self.path: deque[int] = deque() # Remaining path
        self.target_room: str | None = None # Final room
        self.move_time: float | None = None # Time edge movement started

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        '''
        Converts YAML (x, y) float coordinates to integer screen coordinates.

        Params:
            x (float): The x coordinate
            y (float): The y coordinate

        Returns:
            tuple[int, int]: The (x, y) coordinates converted to integers
        '''
        return int(x), int(y)

    def _render_bg(self) -> np.ndarray:
        '''
        Draws the graph, vertices, and edges onto the floor map

        Returns:
            background: The floor plan image containing the graph and notated room numbers
        '''
        background = self.background.copy()

        # Draw graph edges
        for start_id, end_id in self.edges:
            start = self.vertices[start_id]
            end = self.vertices[end_id]

            start_point = self._to_screen(start.x, start.y)
            end_point = self._to_screen(end.x, end.y)

            cv2.line(
                background,
                start_point,
                end_point,
                (180, 180, 180),
                3
            )

        # Draw all vertices
        for vertex in self.vertices.values():
            point = self._to_screen(vertex.x, vertex.y)

            cv2.circle(
                background,
                point,
                5,
                (0, 0, 0),
                -1
            )

        # Identify rooms
        for label, vtx_id in self.rooms.items():
            vtx = self.vertices[vtx_id]

            point = self._to_screen(vtx.x, vtx.y)

            cv2.putText(
                background,
                label,
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return background

    def find_path(self, start_vtx: int, target_vtx: int) -> list[int] | None:
        '''
        Finds a path between two vertices using breadth first search (BFS)

        Params:
            start_vtx (int): The vertex to start from
            target_vtx (int): The vertex to reach

        Returns:
            path (list[int] | None): The vertices passed in order from start to target (None if no path)
        '''
        if start_vtx == target_vtx:
            return [start_vtx]

        queue = deque([start_vtx])
        # Each vertex references what came before it (except the 1st)
        previous: dict[int, int | None] = {start_vtx: None}

        while queue: # While there's vertices left to explore
            current = queue.popleft()

            # Look at all neighbors of current vertex
            for neighbor in self.adjacency[current]:
                if neighbor in previous:
                    continue

                previous[neighbor] = current

                if neighbor == target_vtx: # Target found, make path
                    path = []
                    vertex = target_vtx

                    while vertex is not None:
                        path.append(vertex) # Go backwards (end to beginning)
                        vertex = previous[vertex]

                    path.reverse() # Reverse so path is beginning to end
                    return path

                queue.append(neighbor) # Consider the neighbor in a later loop

        return None

    def move(self, room: str) -> bool:
        '''
       Updates the robot's location to a recognized room

       If this is the first recognized room (beginning of the video), the robot is placed there immediately.
       If not, BFS finds the shortest path to the room.

       Note: This function does NOT animate the movement, that is done by update_position

        Params:
            room (str): The room to be moved to

        Returns:
            bool: Whether the robot successfully moved or not
        '''        
        vertex_id = self.rooms.get(room.upper()) # Vertex belonging to room

        if vertex_id is None:
            return False

        # First recognized room
        if self.curr_vtx is None:
            vertex = self.vertices[vertex_id]
            self.curr_vtx = vertex_id
            self.curr_room = room
            self.curr_point = (vertex.x, vertex.y)
            return True

        # Already at this room & not traveling
        if vertex_id == self.curr_vtx and not self.path:
            self.curr_room = room
            self.curr_point = (self.vertices[vertex_id].x, self.vertices[vertex_id].y)
            return True

        # Don't start 2nd movement while current one still in progress
        if self.path:
            return False

        # Need to travel somewhere else
        path = self.find_path(self.curr_vtx, vertex_id)

        if path is None:
            return False

        # No movement needed
        if len(path) < 2:
            return True

        self.path = deque(path) # Route that needs to be taken
        self.target_room = room # Room that needs to be reached
        self.move_time = time.perf_counter()
        return True

    def _update_position(self):
        '''
        Updates the robot's position along the path based on elapsed time.

        The robot moves at constant speed between vertices.
        '''
        # No path or no move time started
        if len(self.path) < 2 or self.move_time is None:
            return
        
        # 1st edge in path: self.path[0] -> self.path[1]
        # self.vertices[] -> Vertex(x, y)
        start_vtx = self.vertices[self.path[0]]
        target_vtx = self.vertices[self.path[1]]

        start_x, start_y = start_vtx.x, start_vtx.y
        target_x, target_y = target_vtx.x, target_vtx.y

        # Distance between start & end vertices (by pixel)
        edge_dist = np.hypot(target_x - start_x, target_y - start_y)

        # Time elapsed since beginning edge traversal
        elapsed = time.perf_counter() - self.move_time
        
        # Percentage of completion (can't be more than 100% complete, use min 1.0 if more)
        progress = min(elapsed * self.move_speed / edge_dist, 1.0) 

        # Interpolation to get current position along edge
        x = start_x + (target_x - start_x) * progress
        y = start_y + (target_y - start_y) * progress
        self.curr_point = (x, y)

        # Current edge traversal complete
        if progress >= 1.0:
            self.curr_point = (target_x, target_y) # Place robot exactly on target vertex
            self.path.popleft() # Pop completed edge's starting vertex

            # Entire path complete
            if len(self.path) < 2:
                self.curr_vtx = self.path[0]
                self.curr_room = self.target_room # Reached target room

                # Clear movement state
                self.path.clear()
                self.target_room = None
                self.move_time = None
                return

            # Reset time for next edge
            self.move_time = time.perf_counter()
            return

    def display(self):
        '''
        Updates & displays the robot's current position on the floor map
        '''  
        self._update_position() # Called every frame to animate movement
        image = self.map.copy()

        if self.curr_point is not None:
            point = self._to_screen(self.curr_point[0], self.curr_point[1])

            # Robot position
            cv2.circle(
                image,
                point,
                10,
                (255, 0, 0),
                -1
            )

            # Show destination room while moving
            display_room = (self.target_room if self.target_room is not None else self.curr_room)

            if display_room is not None:
                cv2.putText(
                    image,
                    display_room,
                    (point[0] + 15, point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

        display = cv2.resize(image, (0, 0), fx=0.6, fy=0.6) # Map too large for screen, resize
        cv2.imshow("Floor Map",display)