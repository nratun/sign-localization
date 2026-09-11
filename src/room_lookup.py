import yaml
import re
from pathlib import Path

# Focus only on floor 3 right now, since that's where I have the most video content
FLOORS_DIR = Path("dataset/floors")
YAML_NAME = "floor-plan.yaml"
FLOOR_NAME = "floor3"

def load_yaml():
    # Find YAML file and return floor 3 config
    yaml_path = FLOORS_DIR / YAML_NAME

    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    return data["levels"][FLOOR_NAME]

def build_rooms(floor_data: dict) -> set[str]:
    rooms = set()

    for vertex in floor_data["vertices"]:
        label = vertex[3]

        if label is None:
            continue

        rooms.add(str(label).upper())
    return rooms

'''
First check for a complete room like "316A"
Next check simple 3 digit room
Number related to lettered room? Search other extractions for valid A-D suffix\
No valid suffix found? Return simple room
No room found? Return none
'''
def find_room(detections: list[dict], rooms: set[str]) -> str | None:
    # First check for a complete room like "316A"
    for detection in detections:
        text = detection["text"].upper()
        match = re.search(r"\d{3}[A-D]?", text)

        if not match:
            continue

        room = match.group()
        if room in rooms:
            return room_num

    # Next check simple 3 digit room
    for detection in detections:
        text = detection["text"].upper()
        match = re.search(r"\d{3}", text)

        # No 3 digit number found
        if not match:
            continue

        room_num = match.group()

        # No valid room found
        if room_num not in rooms:
            continue

        # Does this number have lettered rooms?
        candidates = [room for room in rooms if room.startswith(room_num) and len(room) == 4]

        # Search other OCR extractions for a valid A-D suffix
        if candidates:
            for candidate in detections:
                suffix = candidate["text"].upper().strip()

                # Candidate doesn't match suffix
                if not re.fullmatch(r"[A-D]", suffix):
                    continue

                candidate = room_num + suffix

                if candidate in rooms:
                    return candidate

        if room_num in rooms:
            return room_num
    return None

def main():
    floor_data = load_yaml()
    rooms = build_rooms(floor_data)

    tests = [
        [{"text": "316A", "conf": 0.99, "points": None}],
        [
            {"text": "316", "conf": 0.95, "points": None},
            {"text": "B", "conf": 0.91, "points": None}
        ],
        [{"text": "335", "conf": 0.99, "points": None}],
        [{"text": "999", "conf": 0.99, "points": None}]
    ]

    for test in tests:
        room = find_room(test, rooms)

        if room:
            print(f"OCR: {test} -> room {room}")
        else:
            print(f"OCR: {test} -> No valid room")


if __name__ == "__main__":
    main()