import math

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polygon
from compas.geometry import Vector
from compas.geometry import Frame
from compas.geometry import Transformation
from compas.geometry import intersection_line_line

from compas.tolerance import TOL
from compas.datastructures import Graph

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
        self.key = None

    @property
    def centroid(self):
        return self.shape.centroid

    @property
    def edges(self):
        return self.shape.lines


    def generate_segments(self):
        self.segments = []
        self.connection_points = []  # Reset connection points

        for edge in self.edges:
            midpoint = edge.midpoint  # Calculate midpoint of the edge

            # Calculate eccentricities
            start = self.centroid + edge.vector * self.start_eccentricity
            end = midpoint + edge.vector * self.end_eccentricity

            # Extend the line inwards
            start = start - Vector.from_start_end(start, end)

            # Create an instance of the RodSegment and append to the correct list
            rod_segment = RodSegment(start, end)
            self.segments.append(rod_segment)

            # Create and store the connection point
            _point, param = edge.closest_point(end, return_parameter=True)
            vector = Vector.from_start_end(start, end).unitized()
            connection = ConnectionPoint(end, param, vector)
            self.connection_points.append(connection)

    def adjust_segments(self, overlap=None):
        overlap = overlap if overlap is not None else self.overlap  # Ensure overlap is not None

        adjusted_segments = []

        num_segments = len(self.segments)
        for i in range(num_segments):
            # Get a pair of segments (current and next, wrapping around)
            segment = self.segments[i]
            next_segment = self.segments[(i + 1) % num_segments]

            # Find the intersection between the current rod and the next rod
            intersection = intersection_line_line((segment.start, segment.end), (next_segment.start, next_segment.end))

            if not intersection or intersection[0] is None:
                print(f"Warning: No intersection found between Rod {i} and Rod {(i + 1) % num_segments}.")
                adjusted_segments.append(segment)  # Keep original rod if no intersection
                continue

            # Use the first intersection point
            intersection_point = Point(*intersection[0])

            # Extend the rod from the intersection point with the specified overlap
            segment_vector = Vector.from_start_end(segment.start, intersection_point).unitized()
            extended_start = intersection_point + segment_vector * -overlap

            # Update the segment's start point
            segment.start = extended_start

            # Store the adjusted rod
            adjusted_segments.append(segment)

        # Replace the old rods with the adjusted ones
        self.segments = adjusted_segments

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RF Grammar Rules lecture
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_edge_frame(self, edge_index):
        point = self.connection_points[edge_index].point
        xaxis = self.edges[edge_index].vector
        yaxis = self.connection_points[edge_index].vector
        frame = Frame(point, xaxis, yaxis)
        return frame

class RFGrammarRule:
    def __init__(self, unit1: RFUnit, unit2: RFUnit = None, edge1: int = 0, edge2: int = 0, mirror: bool = False):
        if unit2 is None:
            unit2 = self.generate_new_unit(unit1, unit1.shape.copy())

        self.unit1 = unit1
        self.unit2 = unit2
        self.edge1 = edge1
        self.edge2 = edge2
        self.mirror = mirror

    def generate_new_unit(self, rfunit: RFUnit, polygon: Polygon) -> RFUnit:
        # Create the new RFUnit using the transformed polygon
        new_unit = RFUnit(
            polygon,
            start_eccentricity=rfunit.start_eccentricity,
            end_eccentricity=rfunit.end_eccentricity,
            overlap=rfunit.overlap,
        )

        # Regenerate segments
        new_unit.generate_segments()
        new_unit.adjust_segments()

        return new_unit

    def apply_rule(self, edge1: int, edge2: int, mirror: bool = False) -> RFUnit:
        frame1 = self.unit1.get_edge_frame(edge1)
        frame2 = self.unit2.get_edge_frame(edge2)

        frame1.yaxis *= -1
        if mirror:
            frame2.xaxis *= -1

        # Transform the polygon and all rods in the unit
        transformation = Transformation.from_frame_to_frame(frame2, frame1)
        transformed_polygon = self.unit2.shape.transformed(transformation)

        # Produce a new unit with the transformed geometry
        new_unit2 = self.generate_new_unit(self.unit2, transformed_polygon)

        return new_unit2

    def reversed(self):
        return RFGrammarRule(self.unit2, self.unit1, self.edge2, self.edge1, mirror=self.mirror)


class RFTessellation:
    def __init__(self, vertex_configuration):
        self.graph = Graph(default_node_attributes={"connected_shape_edges": []})
        self.vertex_configuration = vertex_configuration

    @property
    def units(self):
        return self.graph.nodes_attribute(name="unit")

    def get_next_edge_index(self, unit, current_vertex):
        # Get the edges that are already used/connected
        already_used_edges = self.graph.node_attribute(unit.key, "connected_shape_edges")

        # Find the edge of new_unit that is connected to the current vertex, and is not already used in a connection
        candidate_edges = []
        for edge_index, edge in enumerate(unit.edges):
            # If edge is not already used..
            if edge_index not in already_used_edges:
                # prepare candidates based on the min distance of either of its ends to the current vertex
                d1 = edge.end.distance_to_point(current_vertex)
                d2 = edge.start.distance_to_point(current_vertex)
                min_distance = min(d1, d2)
                candidate_edge_data = (edge_index, min_distance)
                candidate_edges.append(candidate_edge_data)

        sorted_candidate_edges = sorted(candidate_edges, key=lambda x: x[1])
        if len(sorted_candidate_edges) > 0:
            edge_index, distance = sorted_candidate_edges[0]
            return edge_index

        return None

    # **********************************************************
    # START OF MAIN TASK WITHOUT TACKLING CHALLENGES
    # This is the simpler solution to the exercise
    # **********************************************************
    def is_simple_closed_loopback(self, new_unit, first_unit):
        # Simple check for loopback against first unit
        if self.graph.number_of_nodes() == len(self.vertex_configuration):
            # Check if any connection points between the units are close enough
            for i, new_point in enumerate(new_unit.connection_points):
                for j, first_point in enumerate(first_unit.connection_points):
                    if TOL.is_close(new_point.point.distance_to_point(first_point.point), 0):
                        self.graph.add_edge(new_unit.key, first_unit.key)
                        self.graph.node_attribute(new_unit.key, "connected_shape_edges").append(i)
                        self.graph.node_attribute(first_unit.key, "connected_shape_edges").append(j)
                        return True
        
        return False

    # [MAIN TASK]
    def apply_single_rule_cycle(self, rule, vertex_index=0):
        first_unit = rule.unit1
        if not self.graph.has_node(first_unit.key):
            first_unit.key = self.graph.add_node(first_unit.key, unit=first_unit)

        vertex = first_unit.shape.points[vertex_index]
        
        # The second unit is disconnected, so, we can choose any edge, we go for the zeroth
        second_unit_side = 0

        MAX_ITERATIONS = 20
        for iteration in range(MAX_ITERATIONS):
            next_edge = self.get_next_edge_index(rule.unit1, vertex)
            if next_edge is None:
                raise Exception(f"Cannot find next edge of Unit={rule.unit1.key}")

            new_unit = rule.apply_rule(next_edge, second_unit_side, rule.mirror)

            new_unit.key = self.graph.add_node(unit=new_unit, connected_shape_edges=[second_unit_side])
            self.graph.node_attribute(rule.unit1.key, "connected_shape_edges").append(next_edge)
            self.graph.add_edge(rule.unit1.key, new_unit.key)

            # Simple way to close the loop, even on the main task, without a full check
            if self.is_simple_closed_loopback(new_unit, first_unit):
                return

            # NOTE: This is the more complete/checker way to close the loop for challenge 01
            # if self.is_closed_loopback(new_unit):
            #     return

            rule.unit2 = new_unit
            rule = rule.reversed()
        
        raise RuntimeError(f"Failed to close tessellation loop after {MAX_ITERATIONS} iterations")

    # **********************************************************
    # END OF MAIN TASK WITHOUT TACKLING CHALLENGES
    # **********************************************************

    # **********************************************************
    # START OF MAIN TASK WITH BOTH CHALLENGES
    # This is the more complete solution to the exercise
    # **********************************************************

    def is_closed_loopback(self, new_unit: RFUnit) -> bool:
        connected_shape_edges = self.graph.node_attribute(new_unit.key, "connected_shape_edges")
        found_closed_loopback = False

        # For each edge in the newly added unit...
        for i in range(len(new_unit.edges)):
            # If the edge is already connected, ignore it
            if i in connected_shape_edges:
                continue
            
            # If it's not yet connected, check if it forms a loopback
            p1 = new_unit.connection_points[i]

            # Collect all connection points from other units
            connection_data = []
            for n, data in self.graph.nodes(data=True):
                unit = data["unit"]
                if unit == new_unit:
                    continue
                for j, p2 in enumerate(unit.connection_points):
                    connection_data.append((unit, j, p2.point.distance_to_point(p1.point)))

            # Sort connection points by distance to p1
            sorted_points = sorted(connection_data, key=lambda d: d[2])

            # Process only the closest point if it's close enough
            if sorted_points:
                closest_unit, j, distance = sorted_points[0]
                if TOL.is_close(distance, 0):
                    self.graph.add_edge(new_unit.key, closest_unit.key)
                    self.graph.node_attribute(new_unit.key, "connected_shape_edges").append(i)
                    self.graph.node_attribute(closest_unit.key, "connected_shape_edges").append(j)
                    found_closed_loopback = True

        return found_closed_loopback


    def create_vertex_pair_iterator(self):
        i = 0
        while True:
            vertex_a = self.vertex_configuration[i % len(self.vertex_configuration)]
            vertex_b = self.vertex_configuration[(i+1) % len(self.vertex_configuration)]
            yield vertex_a, vertex_b
            i += 1

    def get_next_vertex_pair(self):
        return next(self.vertex_pairs)


    # [MAIN TASK including CHALLENGE 01 and 02]
    def apply_rules_cycle(self, rules, vertex_index=0):
        first_unit = rules[0].unit1
        if not self.graph.has_node(first_unit.key):
            first_unit.key = self.graph.add_node(first_unit.key, unit=first_unit)
        vertex = first_unit.shape.points[vertex_index]

        available_rules = {}
        for rule in rules:
            a = len(rule.unit1.edges)
            b = len(rule.unit2.edges)
            available_rules[a, b] = rule
            if a != b:
                available_rules[b, a] = rule.reversed()

        self.vertex_pairs = self.create_vertex_pair_iterator()
        vertex_pair = self.get_next_vertex_pair()
        rule = available_rules[vertex_pair]

        # The second unit is disconnected, so, we can choose any edge, we go for the zeroth one
        second_unit_side = 0

        MAX_ITERATIONS = 20
        for iteration in range(MAX_ITERATIONS):
            next_edge = self.get_next_edge_index(rule.unit1, vertex)
            print(f"Finding next edge of {rule.unit1.key}, was next_edge={next_edge}")
            if next_edge is None:
                raise Exception(f"Cannot find next edge of Unit={rule.unit1.key}")

            new_unit = rule.apply_rule(next_edge, second_unit_side, rule.mirror)

            new_unit.key = self.graph.add_node(unit=new_unit, connected_shape_edges=[second_unit_side])
            self.graph.node_attribute(rule.unit1.key, "connected_shape_edges").append(next_edge)
            self.graph.add_edge(rule.unit1.key, new_unit.key)

            if self.is_closed_loopback(new_unit):
                return
            
            prev_vertex_pair = vertex_pair
            vertex_pair = self.get_next_vertex_pair()
            if vertex_pair == prev_vertex_pair:
                rule.unit2 = new_unit
                rule = rule.reversed()
            else:
                rule = available_rules[vertex_pair]
                rule.unit1 = new_unit

            if len(new_unit.edges) != vertex_pair[0]:
                raise ValueError(f"Rule mismatch: unit has {len(new_unit.edges)} edges but vertex pair expects {vertex_pair[0]}")
        
        raise RuntimeError(f"Failed to close tessellation loop after {MAX_ITERATIONS} iterations")

    # **********************************************************
    # END OF MAIN TASK WITH BOTH CHALLENGES
    # **********************************************************