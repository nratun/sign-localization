import yaml
from pathlib import Path

# Focus only on floor 3 right now, since that's where I have the most video content
FLOORS_DIR = Path("dataset/floors")
YAML_NAME = "floor-plan.yaml"
FLOOR_NAME = "floor3"

# Vertices that I care about for the sake of testing
LANDMARKS = {
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
    "335",
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
        vertices: vertex_id -> { "x": x, "y": y, "label": label}
        edges: [(start_id, end_id), ...]
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

# Used to be filter_useful_vertices but the name was ugly
def get_landmarks(vertices):
    # Extract only useful vertices, but keep og vertex id
    landmarks = {}

    for vertex_id, vertex in vertices.items():
        label = vertex["label"]

        if label in LANDMARKS:
            landmarks[label] = {
                "vertex_id": vertex_id,
                "x": vertex["x"],
                "y": vertex["y"]
            }

    return landmarks

def main():
    floor_data = load_yaml()
    vertices, edges = build_graph(floor_data)
    landmarks = get_landmarks(vertices)

    curr_vtx = None
    # Test OCR outputs
    test_texts = ["316a", "316 B", "316C.", "317D", "999", "", "CAR LAB 316 X-RAY", "ROOM 316B"]

if __name__ == "__main__":
    main()