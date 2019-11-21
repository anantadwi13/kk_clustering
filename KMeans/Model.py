from typing import List


class Point:
    def __init__(self, name: str, pos: List[float]):
        self.name = name
        self.pos = pos


class Centroid:
    def __init__(self, name: str, pos: List[float], members: List[Point]):
        self.name = name
        self.pos = pos
        self.members = members

