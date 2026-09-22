import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# Focus only on floor 3 right now, since that's where I have the most video content
FLOORS_DIR = Path("dataset/floors")
YAML_NAME = "floor-plan.yaml"
FLOOR_NAME = "floor3"
MOVE_SPEED = 200.0

@dataclass
class Vertex:
    x: float
    y: float
    label: str | None = None

@dataclass
class Room:
    vertex_id: int

def load_yaml():
    # Find YAML file and return floor 3 config
    yaml_path = FLOORS_DIR / YAML_NAME

    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    return data["levels"][FLOOR_NAME]

def build_graph(floor_data: dict) -> tuple[dict[int, Vertex], list[tuple[int, int]]]:
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


def get_rooms(vertices: dict[int, Vertex]) -> dict[str, Room]:
    rooms = {}

    for vertex_id, vertex in vertices.items():
        if not vertex.label:
            continue

        rooms[vertex.label] = Room(vertex_id=vertex_id)
    return rooms


class Map:
    def __init__(
        self,
        floor_data: dict,
        vertices: dict[int, Vertex],
        edges: list[tuple[int, int]],
        rooms: dict[str, Room],
        move_speed: float = MOVE_SPEED
    ):
        self.vertices = vertices
        self.edges = edges
        self.rooms = rooms
        self.move_speed = move_speed

        # TODO change name from drawing & background to something else?
        drawing_name = floor_data["drawing"]["filename"]
        drawing_path = FLOORS_DIR / drawing_name

        self.background = cv2.imread(str(drawing_path))

        if self.background is None:
            raise FileNotFoundError(f"Could not load floor plan: {drawing_path}")

        # Render map once
        self.map = self._render_bg()

        # Build adjacency list for graph traversal
        self.adjacency: dict[int, list[int]] = {vertex_id: [] for vertex_id in self.vertices}

        for start_id, end_id in self.edges:
            self.adjacency[start_id].append(end_id)
            self.adjacency[end_id].append(start_id)
        # ---------------------- Robot Location ----------------------
        self.curr_vtx: int | None = None # Last confirmed room
        self.curr_room: str | None = None # Last confirmed room
        self.curr_point: tuple[float, float] | None = None # Current robot position
        self.path: deque[int] = deque() # Remaining path
        self.target_room: str | None = None # Final room
        self.move_time: float | None = None

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        # YAML coords to screen coords
        return int(x), int(y)

    def _render_bg(self) -> np.ndarray:
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
                2
            )

        # Draw all vertices
        for vertex in self.vertices.values():
            point = self._to_screen(vertex.x, vertex.y)

            cv2.circle(
                background,
                point,
                4,
                (0, 0, 0),
                -1
            )

        # Identify rooms
        for label, room in self.rooms.items():
            vertex = self.vertices[room.vertex_id]

            point = self._to_screen(vertex.x, vertex.y)

            cv2.putText(
                background,
                label,
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return background

    def room_to_vtx(self, room: str) -> int | None:
        room = self.rooms.get(room.upper())

        if room is None:
            return None

        return room.vertex_id

    def find_path(self, start_vtx: int, target_vtx: int) -> list[int] | None:
        if start_vtx == target_vtx:
            return [start_vtx]

        queue = deque([start_vtx])
        previous: dict[int, int | None] = {start_vtx: None}

        while queue:
            current = queue.popleft()

            for neighbor in self.adjacency[current]:
                if neighbor in previous:
                    continue

                previous[neighbor] = current

                if neighbor == target_vtx:
                    path = []
                    vertex = target_vtx

                    while vertex is not None:
                        path.append(vertex)
                        vertex = previous[vertex]

                    path.reverse()
                    return path

                queue.append(neighbor)

        return None

    def move(self, room: str) -> bool:
        vertex_id = self.room_to_vtx(room)

        if vertex_id is None:
            print(f"[MAP] Room {room} not found in rooms")
            return False

        # First recognized room
        if self.curr_vtx is None:
            vertex = self.vertices[vertex_id]
            self.curr_vtx = vertex_id
            self.curr_room = room
            self.curr_point = (vertex.x, vertex.y)

            print(f"[MAP] Initial position: {room} -> vertex {vertex_id}")
            return True

        # Already at this room
        if vertex_id == self.curr_vtx and not self.path:
            self.curr_room = room
            self.curr_point = (self.vertices[vertex_id].x, self.vertices[vertex_id].y)
            return True

        # Don't start 2nd movement while current one still in progress
        if self.path:
            print(f"[MAP] Already moving toward {self.target_room}; ignoring {room}")
            return False

        path = self.find_path(self.curr_vtx, vertex_id)

        if path is None:
            print(f"[MAP] No path from vertex {self.curr_vtx} to vertex {vertex_id}")
            return False

        if len(path) < 2:
            return True

        self.path = deque(path)
        self.target_room = room
        self.move_time = time.perf_counter()

        print(
            f"[MAP] Moving {self.curr_vtx} -> "
            f"{room} via {path}"
        )

        return True

    def _update_position(self):
        if len(self.path) < 2 or self.move_time is None:
            return

        start_vtx = self.vertices[self.path[0]]
        target_vtx = self.vertices[self.path[1]]
        start_x, start_y = start_vtx.x, start_vtx.y
        target_x, target_y = target_vtx.x, target_vtx.y

        edge_dist = np.hypot(target_x - start_x, target_y - start_y)

        if edge_dist <= 0:
            progress = 1.0
        else:
            elapsed = time.perf_counter() - self.move_time
            progress = min(elapsed * self.move_speed / edge_dist, 1.0)

        x = start_x + (target_x - start_x) * progress
        y = start_y + (target_y - start_y) * progress
        self.curr_point = (x, y)

        # Current edge complete
        if progress >= 1.0:
            self.curr_point = (target_x, target_y)
            self.path.popleft()

            # Entire path complete
            if len(self.path) < 2:
                self.curr_vtx = self.path[0]
                self.curr_room = self.target_room
                self.path.clear()
                self.target_room = None
                self.move_time = None

                print(f"[MAP] Arrived at {self.curr_room}")
                return

            self.move_time = time.perf_counter()
            return

    def display(self):
        self._update_position()
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
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

        display = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow("Floor Map",display)