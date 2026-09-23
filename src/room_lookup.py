#!/usr/bin/env python

"""
room_lookup.py: Compares OCR text extracted from a building sign with valid rooms in a floor plan's YAML
"""
import re


def find_room(detections: list[dict], rooms: dict[str, int]) -> str | None:
    '''
    Takes in a list of extracted OCR text and a set of room numbers.
    Returns a valid room number identified in the OCR text (None if no valid room is extracted).

    Params:
        detections (list[dict]): The extracted OCR data ({text, conf})
        rooms (dict[str, int]): The valid rooms to compare the extracted text against

    Returns:
        str | None: The valid room number (None if no valid room)
    '''
    # 1. Check for a complete room (Ex. "316A")
    for detection in detections:
        text = detection["text"].upper().strip()
        normalize = re.sub(r"[\s\-]", "", text)
        match = re.search(r"\d{3}[A-E]", normalize)

        # No complete room found
        if not match:
            continue

        room = match.group()
        if room in rooms:
            return room

    # 2. Check for a simple 3 digit room (Ex. 397)
    for detection in detections:
        text = detection["text"].upper()
        match = re.search(r"\d{3}", text)

        # No 3 digit number found
        if not match:
            continue

        room_num = match.group()
        
        # Does this number have lettered rooms?
        candidates = [room for room in rooms if room.startswith(room_num) and len(room) == 4]

        # Yes -> Search other OCR extractions for a valid A-D suffix
        if candidates:
            for candidate in detections:
                suffix = candidate["text"].upper().strip()

                # Candidate suffix doesn't match
                if not re.fullmatch(r"[A-E]", suffix):
                    continue

                candidate = room_num + suffix

                # Check if the suffixed room exists
                if candidate in rooms:
                    return candidate

        # No -> Check if the 3 digit room exists
        if room_num in rooms:
            return room_num

    # Room does not exist
    return None