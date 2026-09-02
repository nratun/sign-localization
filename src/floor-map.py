import yaml
import re
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

'''
Normalize OCR text so it can be compared with room labels.
Examples:
    "316a" -> "316A"
    "316 A" -> "316A"
    "316A." -> "316A"
    " 316A " -> "316A"
'''
def fix_text(text):
    # This would not work in the event that the ocr can't detect 3 numbers or mistakes a number for a letter
    if not text: return ""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text) # Remove spaces & unnecessary stuff
    match = re.search(r"\d{3}[A-Z]?", text) # Find 3 digits & letter
    if not match: return ""
    return match.group()

'''Create a dictionary mapping room labels to their vertices'''
def map_rooms(landmarks):
    rooms = {}

    for label, landmark in landmarks.items():
        normalized = fix_text(label)

        rooms[normalized] = { 
            "vertex_id": landmark["vertex_id"],
            "x": landmark["x"],
            "y": landmark["y"],
            "label": label
        }
    return rooms

'''
Match OCR text against known room labels.
Returns: Matched room dictionary, or None if no match exists
'''
def match_room(text, room_lookup):
    normalized = fix_text(text)
    if not normalized: return None
    return room_lookup.get(normalized)

def main():
    floor_data = load_yaml()
    vertices, edges = build_graph(floor_data)
    landmarks = get_landmarks(vertices)
    room_lookup = map_rooms(landmarks)

    # Test OCR outputs
    test_texts = ["316a", "316 B", "316C.", "317D", "999", "", "CAR LAB 316 X-RAY", "ROOM 316B"]

    for text in test_texts:
        match = match_room(text, room_lookup)

        if match:
            print (
                f"OCR: '{text}' -> "
                f"Room: {match['label']} "
                f"(vertex {match['vertex_id']}, "
                f"x={match['x']:.1f}, y={match['y']:.1f})"
            )
        else:
            print(f"OCR: '{text}' -> No match")

if __name__ == "__main__":
    main()