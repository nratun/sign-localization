#!/usr/bin/env python

"""
floor_data.py: Loads and extracts floor plan data from a YAML

This file contains central content that is used by both room_lookup.py and map.py
"""
from pathlib import Path

import yaml

# Focusing only on floor 3 for the purpose of demonstration
FLOORS_DIR = Path("dataset/floors")
YAML_NAME = "floor-plan.yaml"
FLOOR_NAME = "floor3"


def load_floor() -> dict:
    '''
    Finds and loads data from a YAML pertaining to a specific building floor

    Returns:
        dict: The information of a particular building floor
    '''
    yaml_path = FLOORS_DIR / YAML_NAME

    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    return data["levels"][FLOOR_NAME]


def build_rooms(floor_data: dict) -> dict[str, int]:
    '''
    Takes in a dictionary containing data about a particular building floor.
    Extracts all named rooms and returns their names & vertex IDs as a dict.

    Params:
        floor_data (dict): The information of a particular building floor

    Returns:
        rooms (dict[str, int]): The numbered/named rooms on a particular floor & their vertex IDs
    '''
    rooms = {}

    for vertex_id, vertex in enumerate(floor_data["vertices"]):
        label = vertex[3]

        if not label:
            continue

        rooms[str(label).upper()] = vertex_id

    return rooms