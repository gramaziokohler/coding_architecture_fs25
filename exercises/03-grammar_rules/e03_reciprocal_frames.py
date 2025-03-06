from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polygon
from compas.geometry import Vector
from compas.geometry import Frame
from compas.geometry import Transformation
from compas.geometry import intersection_line_line

class ConnectionPoint:
    def __init__(self, point: Point, parameter: float, vector: Vector):
        self.point = point
        self.parameter = parameter
        self.vector = vector

class RodSegment:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end
    
    @property
    def line(self) -> Line:
        return Line(self.start, self.end)

class RFUnit:
    def __init__(self, shape: Polygon, start_eccentricity: float, end_eccentricity: float, overlap: float):
        self.shape = shape
        self.start_eccentricity = start_eccentricity
        self.end_eccentricity = end_eccentricity
        self.overlap = overlap
        self.connection_points = []
        self.segments = []

    @property
    def centroid(self):
        return self.shape.centroid

    @property
    def edges(self):
        return self.shape.lines

    def generate_segments(self):
        # ...
        pass


    def adjust_segments(self):
        # ...
        pass


class RFGrammarRule:
    def __init__(self, unit1, unit2, edge1, edge2, mirror):
        # ...
        pass
