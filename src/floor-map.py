import yaml
import cv2
from pathlib import Path

# Focus only on floor 3 right now, since that's where I have the most video content
FLOORS_DIR = Path("dataset/floors")
YAML_NAME = "floor-plan.yaml"
FLOOR_NAME = "floor3"
IMAGE_NAME = "FinTech_floor_3-1.png"

# Vertices that I care about for the sake of testing
USEFUL_LABELS = {
    "303",
    "305",
    "306",
    "315",
    "316",
    "316A",
    "316B",
    "316C",
    "316D",
    "317",
    "320",
    "321",
    "322",
    "323",
    "324",
    "325",
    "326",
    "327",
    "328",
    "331",
    "335_kitchen",
    "336",
}

def load_yaml():
    #Find YAML file and return floor 3 config
    yaml_path = FLOORS_DIR / YAML_NAME

    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    return data["levels"][FLOOR_NAME]

def build_graph(yaml):
    #Convert YAML graph into Python
    # Keep unlabeled vertices as well
    '''
    Returns:

        vertices:
            vertex_id -> {
                "x": x,
                "y": y,
                "label": label
            }

        edges:
            [(start_id, end_id), ...]
    '''

    vertices = {}

    for vertex_id, vertex in enumerate(yaml["vertices"]):
        x, y, label = vertex[0], vertex[1], vertex[3]

        # Convert all labels to strings
        if label is not None:
            label = str(label)

        vertices[vertex_id] = {
            "x": x,
            "y": y,
            "label": label
        }

    edges = []

    for edge in yaml["lanes"]:
        start = edge[0]
        end = edge[1]
        edges.append((start, end))

    return vertices, edges

def filter_useful_vertices(vertices):
    # Extract only useful vertices, but keep og vertex id
    useful_vertices = {}

    for vertex_id, vertex in vertices.items():
        label = vertex["label"]
        if label in USEFUL_LABELS:
            useful_vertices[vertex_id] = vertex

    return useful_vertices

def draw_graph(
    image,
    vertices,
    edges,
    useful_vertices
):
    # Draw navigation graph with useful vertices
    # All edges are visible, but nly useful vertices are labeled

    output = image.copy()

    for start_id, end_id in edges:
        start = vertices[start_id]
        end = vertices[end_id]

        p1 = (int(start["x"]), int(start["y"]))

        p2 = (int(end["x"]), int(end["y"]))

        cv2.line(
            output,
            p1,
            p2,
            (0, 255, 0),
            2
        )

    # Draw useful vertices only
    for vertex_id, vertex in useful_vertices.items():
        x = int(vertex["x"])
        y = int(vertex["y"])

        # Landmark point
        cv2.circle(
            output,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

        # Landmark label
        label = vertex["label"]

        cv2.putText(
            output,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    return output

def main():
    floor_data = load_yaml()
    vertices, edges = build_graph(floor_data)
    useful_vertices = filter_useful_vertices(vertices)

    # Load floor plan img
    img_path = FLOORS_DIR / IMAGE_NAME
    img = cv2.imread(str(img_path))

    if img is None:
        raise FileNotFoundError(
            f"Could not load floor plan image: {img_path}"
        )

    output = draw_graph(
        img,
        vertices,
        edges,
        useful_vertices
    )

    # Resize
    scale = 0.75
    output = cv2.resize(
        output,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(
        "Floor 3 Graph",
        output
    )

    print("\nPress any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()