from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# Focus only on floor 3 right now, since that's where I have the most video content
FLOORS_DIR = Path("dataset/floors")
YAML_NAME = "floor-plan.yaml"
FLOOR_NAME = "floor3"

@dataclass
class Vertex:
    x: float
    y: float
    label: str | None = None


@dataclass
class Landmark:
    vertex_id: int

def load_yaml():
    #Find YAML file and return floor 3 config
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


def get_landmarks(vertices: dict[int, Vertex]) -> dict[str, Landmark]:
    # Extract only useful vertices, but keep og vertex id
    landmarks = {}

    for vertex_id, vertex in vertices.items():
        if not vertex.label:
            continue

        landmarks[vertex.label] = Landmark(vertex_id=vertex_id)

    return landmarks

class FloorMap:
    def __init__(
        self,
        floor_data: dict,
        vertices: dict[int, Vertex],
        edges: list[tuple[int, int]],
        landmarks: dict[str, Landmark],
    ):
        self.vertices = vertices
        self.edges = edges
        self.landmarks = landmarks

        drawing_name = floor_data["drawing"]["filename"]
        drawing_path = FLOORS_DIR / drawing_name

        self.background = cv2.imread(str(drawing_path))
        # Render floor plan map only once
        self.map = self._render_background()

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        # YAML coords turn to screen coords
        return int(x), int(y)

    def _render_background(self) -> np.ndarray:
        background = self.background.copy()

        # Draw graph edges.
        for start_id, end_id in self.edges:
            start = self.vertices[start_id]
            end = self.vertices[end_id]

            start_point = self._to_screen(start.x, start.y)
            end_point = self._to_screen(end.x, end.y)

            cv2.line(background, start_point, end_point, (180, 180, 180), 2)

        # Draw all vertices
        for vertex in self.vertices.values():
            point = self._to_screen(vertex.x, vertex.y)

            cv2.circle(background, point, 4, (0, 0, 0), -1)

        # Identify landmarks
        for label, landmark in self.landmarks.items():
            vertex = self.vertices[landmark.vertex_id]
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

    def room_to_vertex(self, room: str) -> int | None:
        landmark = self.landmarks.get(room.upper())

        if landmark is None:
            return None

        return landmark.vertex_id

    def display(self, current_vertex: int | None = None, current_room: str | None = None):
        # Display everything (floor plan, robot, etc.)
        image = self.map.copy()

        if current_vertex is not None:
            vertex = self.vertices.get(current_vertex)

            if vertex is not None:
                point = self._to_screen(vertex.x, vertex.y)

                # Robot position.
                cv2.circle(image, point, 10, (255, 0, 0), -1)

                # Current room label
                if current_room is not None:
                    cv2.putText(
                        image,
                        current_room,
                        (point[0] + 15,point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA
                    )

        display = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow("Floor Map", display)

def main():
    floor_data = load_yaml()
    vertices, edges = build_graph(floor_data)
    landmarks = get_landmarks(vertices)

    floor_map = FloorMap(floor_data, vertices, edges, landmarks)

    curr_vtx = None
    curr_room = None
    # Test OCR outputs
    test_rooms = [ "316A", "316B", "317", "335", "303"]

    for room in test_rooms:
        vertex = floor_map.room_to_vertex(room)

        if vertex is not None:
            curr_room = room
            curr_vtx = vertex

        floor_map.display(curr_vtx, curr_room)

        key = cv2.waitKey(1000) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()