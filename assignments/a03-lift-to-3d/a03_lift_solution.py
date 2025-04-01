"""
ASSIGNMENT 03:

The majority of the code in this file is based on the assignment A01 and A02 solutions.

Your task is to add methods and/or functions to the `RFTessellation` class to complete
the tasks of this assignment. The beginning of the relevant section and the line-by-line guidance
for this assignment is marked with "BEGIN: ASSIGNMENT 03" and "END: ASSIGNMENT 03" towards
the end of this file.
"""
import math
from typing import Iterator

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polygon
from compas.geometry import Vector
from compas.geometry import Frame
from compas.geometry import Transformation
from compas.geometry import Translation
from compas.geometry import intersection_line_line

from compas.tolerance import TOL
from compas.datastructures import Graph

from compas_timber.connections import XLapJoint
from compas_timber.elements import Beam
from compas_timber.model import TimberModel


class ConnectionPoint:
    def __init__(self, point: Point, parameter: float, vector: Vector):
        self.point = point
        self.parameter = parameter
        self.vector = vector


class RodSegment:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end
        self.rod = None  # NEW

    @property
    def line(self) -> Line:
        return Line(self.start, self.end)


class Rod:
    def __init__(self, segment1: RodSegment, segment2: RodSegment = None):
        self.key = None
        self.segment1 = segment1
        self.segment2 = segment2
        self.beam = None

    @property
    def line(self) -> Line:
        if self.segment2 is not None:
            return Line(self.segment1.start, self.segment2.start)
        else:
            return Line(self.segment1.start, self.segment1.end)


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
    def centroid(self) -> Point:
        return self.shape.centroid

    @property
    def edges(self) -> list[Line]:
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

    def adjust_segments(self, overlap: float = None):
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

    def get_edge_frame(self, edge_index: int) -> Frame:
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

    def reversed(self) -> "RFGrammarRule":
        return RFGrammarRule(self.unit2, self.unit1, self.edge2, self.edge1, mirror=self.mirror)

    def copy(self) -> "RFGrammarRule":
        return RFGrammarRule(self.unit1, self.unit2, self.edge1, self.edge2, mirror=self.mirror)


class RFTessellation:
    def __init__(self, vertex_configuration: list[float]):
        self.graph = Graph(default_node_attributes={"connected_shape_edges": []})
        self.vertex_configuration = vertex_configuration

    @property
    def units(self) -> list[RFUnit]:
        return self.graph.nodes_attribute(name="unit")

    def get_next_edge_index(self, unit: RFUnit, current_vertex: Point) -> int:
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
            for unit in self.units:
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

    def create_vertex_pair_iterator(self) -> Iterator[tuple[int, int]]:
        i = 0
        while True:
            vertex_a = self.vertex_configuration[i % len(self.vertex_configuration)]
            vertex_b = self.vertex_configuration[(i + 1) % len(self.vertex_configuration)]
            yield vertex_a, vertex_b
            i += 1

    def get_next_vertex_pair(self) -> tuple[int, int]:
        return next(self.vertex_pairs)

    def apply_rules_cycle(self, rules: list[RFGrammarRule], vertex_index: int = 0, first_unit: RFUnit = None) -> list[RFUnit]:
        first_unit = first_unit or rules[0].unit1
        if not self.graph.has_node(first_unit.key):
            first_unit.key = self.graph.add_node(first_unit.key, unit=first_unit)
        vertex = first_unit.shape.points[vertex_index]

        available_rules = {}
        for rule in rules:
            a = len(rule.unit1.edges)
            b = len(rule.unit2.edges)
            available_rules[a, b] = rule.copy()
            if a != b:
                available_rules[b, a] = rule.reversed()

        # Find first matching vertex pair according to the input first unit
        self.vertex_pairs = self.create_vertex_pair_iterator()

        for i in range(10):
            vertex_pair = self.get_next_vertex_pair()

            # Skip vertex pair if first unit has different edge count
            if vertex_pair[0] != len(first_unit.edges):
                continue

            # Found next vertex pair
            rule = available_rules[vertex_pair]
            rule.unit1 = first_unit
            break

        if vertex_pair[0] != len(rule.unit1.edges):
            raise Exception(f"The selected first unit does not match the expected number of edges (expected={vertex_pair[0]}, actual={len(rule.unit1.edges)})")

        # The second unit is disconnected, so, we can choose any edge, we go for the zeroth one
        second_unit_side = 0
        new_units = []

        MAX_ITERATIONS = 20
        for iteration in range(MAX_ITERATIONS):
            next_edge = self.get_next_edge_index(rule.unit1, vertex)
            if next_edge is None:
                continue

            new_unit = rule.apply_rule(next_edge, second_unit_side, rule.mirror)
            new_unit.key = self.graph.add_node(unit=new_unit, connected_shape_edges=[second_unit_side])
            self.graph.node_attribute(rule.unit1.key, "connected_shape_edges").append(next_edge)
            self.graph.add_edge(rule.unit1.key, new_unit.key)

            new_units.append(new_unit)

            if self.is_closed_loopback(new_unit):
                return new_units

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

    def find_next_unit_with_free_edge(self, all_units: list[RFUnit]) -> RFUnit:
        MAX_ATTEMPTS = 30

        for _ in range(MAX_ATTEMPTS):
            if not all_units:
                return None

            unit = all_units[0]
            connected_edges = self.graph.node_attribute(unit.key, "connected_shape_edges")
            free_edges = [i for i in range(len(unit.edges)) if i not in connected_edges]

            if free_edges:
                return unit, free_edges[0]

            # No free edges, remove from list and try next
            all_units.pop(0)

        return None

    def tessellate(self, rules: list[RFGrammarRule], max_iterations: int = 3) -> None:
        # Initial rule application
        vertex_index = 0
        first_unit = None

        # Find first unit with matching edge count according to the vertex configuration
        for rule in rules:
            if len(rule.unit1.edges) == self.vertex_configuration[0]:
                first_unit = rule.unit1
                break

        all_units = [first_unit]
        new_units = self.apply_rules_cycle(rules, vertex_index, first_unit)
        all_units.extend(new_units)

        for _ in range(max_iterations):
            print(f"Iteration {_}")
            next_unit_info = self.find_next_unit_with_free_edge(all_units)
            if not next_unit_info:
                break
            unit, vertex_index = next_unit_info

            new_units = self.apply_rules_cycle(rules, vertex_index, unit)
            all_units.extend(new_units)

        # Assign xyz to graph nodes
        for unit in self.units:
            self.graph.node_attributes(unit.key, "xyz", list(unit.centroid))

    def copy(self) -> "RFTessellation":
        copied = RFTessellation(self.vertex_configuration[:])
        copied.graph = self.graph.copy()
        return copied

    @property
    def rods(self) -> list[Rod]:
        seen = set()
        ordered_rods = []
        for unit in self.units:
            for seg in unit.segments:
                if seg.rod and seg.rod not in seen:
                    seen.add(seg.rod)
                    ordered_rods.append(seg.rod)
        
        return ordered_rods

    def create_rods_from_single_segments(self):
        for unit in self.units:
            for seg in unit.segments:
                new_rod = Rod(segment1=seg)
                seg.rod = new_rod

    def create_rods_from_shared_segments(self, angular_tolerance: float = 5.0) -> None:
        # Track segments that have already been assigned to a rod
        processed_segments = set()
        
        # Process each unit and its segments
        for unit in self.units:
            for i in range(len(unit.segments)):
                matching_segment = None
                segment = unit.segments[i]
                connection_point = unit.connection_points[i]

                # Skip if segment already processed
                if segment in processed_segments:
                    continue
                
                # Get neighboring units from graph
                neighbors = self.graph.neighbors(unit.key)
                
                # Check each neighbor for a matching segment
                matching_segment = None
                neighbor_unit = None
                for neighbor_key in neighbors:
                    neighbor_unit = self.graph.node_attribute(neighbor_key, "unit")

                    # Check each segment in neighbor
                    for j in range(len(neighbor_unit.segments)):
                        nbr_segment = neighbor_unit.segments[j]
                        nbr_connection_point = neighbor_unit.connection_points[j]

                        # Skip if neighbor segment already processed
                        if nbr_segment in processed_segments:
                            continue
                            
                        # Check if connection points match
                        if TOL.is_allclose(connection_point.point, nbr_connection_point.point):
                            # Check collinearity
                            vec1 = segment.line.vector.unitized()
                            vec2 = nbr_segment.line.vector.unitized()
                            
                            angle = vec1.angle(vec2)
                            if angle >= angular_tolerance:
                                continue

                            # We found a matching segment
                            matching_segment = nbr_segment
                            break
                    
                    if matching_segment:
                        break

                # Create appropriate rod
                if matching_segment:
                    # Create merged rod from both segments
                    new_rod = Rod(segment, matching_segment)
                    segment.rod = new_rod
                    matching_segment.rod = new_rod
                    processed_segments.add(segment)
                    processed_segments.add(matching_segment)
                else:
                    # Create single-segment rod
                    new_rod = Rod(segment)
                    segment.rod = new_rod
                    processed_segments.add(segment)

    def generate_beams(self, width, height, z_vector=None, surface=None, start_extension=0.0, end_extension=0.0):
        model = TimberModel()

        for rod in self.rods:
            if surface is not None:
                _, uv_param = surface.closest_point(rod.line.midpoint, return_parameters=True)
                z_vector = surface.frame_at(*uv_param).normal

            # Create a beam from the rod's line.
            beam = Beam.from_centerline(centerline=rod.line, width=width, height=height, z_vector=z_vector)

            # Optionally extend the beam (if desired)
            if start_extension or end_extension:
                beam.add_blank_extension(start=start_extension, end=end_extension)

            # Assign the created beam to the rod.
            rod.beam = beam
            beam.attributes["rod"] = rod

            # Add the beam to the timber model
            model.add_element(beam)

        return model

    def generate_joints(self, model, flip_toggle, cut_plane_bias=0.5):
        # For each unit, gather beams
        for unit in self.units:
            beams = []
            
            for seg in unit.segments:
                if seg.rod and seg.rod.beam:
                    beams.append(seg.rod.beam)
                    # vectors.append(seg.rod.beam.centerline.vector)

            num_beams = len(beams)
            if num_beams < 2:
                print(f"RFUnit {unit.key} has less than 2 beams. Skipping joint creation.")
                continue

            # Create joints between consecutive beams (wrapping around)
            for i in range(num_beams):
                j = (i + 1) % num_beams
                beam_a = beams[i]
                beam_b = beams[j]

                dot = beam_a.centerline.vector.dot(beam_b.centerline.vector)
                flip_side = dot > 0.0

                if flip_toggle:
                    flip_side = not flip_side

                XLapJoint.create(model, beam_a, beam_b, flip_side, cut_plane_bias)


    def generate_joints_mapped_tessellation(self, model, flip_toggle, cut_plane_bias=0.5):
        # output lists for testing
        intersection_points1 = []
        intersection_points2 = []
        plane_cut_vectors = []
        offset_vectors = []

        # For each unit, gather beams
        for unit in self.units:
            beams = []
            
            for seg in unit.segments:
                if seg.rod and seg.rod.beam:
                    beams.append(seg.rod.beam)

            num_beams = len(beams)
            if num_beams < 2:
                print(f"RFUnit {unit.key} has less than 2 beams. Skipping joint creation.")
                continue

            # Create joints between consecutive beams (wrapping around)
            for i in range(num_beams):
                j = (i + 1) % num_beams
                beam_a = beams[i]
                beam_b = beams[j]

                # # for planar tessellation
                # dot = beam_a.centerline.vector.dot(beam_b.centerline.vector)
                # flip_side = dot > 0.0

                # Get centerlines as COMPAS Lines
                line_a = beam_a.centerline
                line_b = beam_b.centerline

                main_beam = beam_a
                cross_beam = beam_b

                # Compute intersection
                int_a, int_b = intersection_line_line(line_a, line_b, tol=0.4)

                point_a = Point(*int_a)
                point_b = Point(*int_b)

                intersection_points1.append(point_a)
                intersection_points2.append(point_b)
                
                print(f"Intersection Points: {point_a, point_b}")

                if point_a is None or point_b is None:
                    print(f"No intersection found between beams {i} and {j}")

                # for mapped tessellation
                if point_a is not None and point_b is not None:
                    
                    print(f"Intersection found between beams {i} and {j}")

                    # Ensure correct ordering of main & cross beams based on Z comparison
                    if point_a.z > point_b.z:
                        main_beam, cross_beam = beam_a, beam_b
                    else:
                        main_beam, cross_beam = beam_b, beam_a

                    # Compute Cut Plane Vector using Cross Product
                    plane_cut_vector = main_beam.centerline.vector.cross(cross_beam.centerline.vector)

                    print(f"Plane Cut Vector: {plane_cut_vector}")

                    # Compute Offset Vector from Intersection
                    offset_vector = Vector.from_start_end(point_a, point_b)

                    # Flip if the vectors are pointing in opposite directions
                    flip_side = plane_cut_vector.dot(offset_vector) < 0

                    print(f"Flip Side: {flip_side}")

                    if flip_toggle:
                        flip_side = not flip_side
                    
                    plane_cut_vectors.append(plane_cut_vector)
                    offset_vectors.append(offset_vector)
                    
                                    
                    XLapJoint.create(model, main_beam, cross_beam, flip_side, cut_plane_bias)

        return intersection_points1, intersection_points2, plane_cut_vectors, offset_vectors

    def generate_beam_stats(self, stock_beams_needed, stock_length, total_waste, lm_price, transport_cost, wood_density):
        beams = [rod.beam for rod in self.rods]

        # Calculate beam statistics
        beam_lengths = [beam.centerline.length for beam in beams]
        total_beams = len(beams)
        total_beam_length = sum(beam_lengths)
        largest_beam = max(beam_lengths) if beam_lengths else 0
        shortest_beam = min(beam_lengths) if beam_lengths else 0
        
        # Calculate volume and weight (approximate)
        # Volume = sum of (length * width * height) for each beam
        total_volume = sum(beam.geometry.volume for beam in beams)
        total_weight = total_volume * wood_density
        
        # Calculate cost
        total_stock_length = stock_beams_needed * stock_length
        timber_cost = total_stock_length * lm_price
        total_cost = timber_cost + transport_cost
        
        # Calculate efficiency
        efficiency = (1 - total_waste / total_stock_length) * 100 if total_stock_length > 0 else 0
        
        # Format the statistics
        stats = f"""
        BEAM STATS
        ----------
        Total beams: {total_beams}
        Total beam length: {total_beam_length:.2f} m
        Largest beam: {largest_beam:.2f} m
        Shortest beam: {shortest_beam:.2f} m
        Total volume: {total_volume:.3f} m³
        Total weight: {total_weight:.1f} kg
        ----------
        Stock beams needed: {stock_beams_needed}
        Total stock length: {total_stock_length:.2f} m
        Timber cost: {timber_cost:.2f} CHF
        Transport cost: {transport_cost:.2f} CHF
        Total cost: {total_cost:.2f} CHF
        ----------
        Total waste material: {total_waste:.2f} m
        Efficiency: {efficiency:.1f}%
        """
        print(stats)

        return stats

    # ***************************************************************************************************
    # 
    # BEGIN: ASSIGNMENT 03
    # This is the starting point for the code that is required to complete the tasks of the assignment 03
    # 
    # ***************************************************************************************************


    # *********************************
    # BEGIN: MAIN TASK
    # *********************************
    def map_to_surface(self, source_surface, target_surface):
        # 1. Create a copy of the tessellation
        # [..ADD YOUR CODE HERE..]

        # 2. For each unit in the (copied) tessellation
        # [..ADD YOUR CODE HERE..]

            # 3. Map each of the points of the unit's shape,from the source to the target surface.
            #    Since the mapping will be reused, we suggest to write a function (outside the RFTessellation class)
            #    that will take a point and the two surfaces, and perform the mapping and return a new point.
            # [..ADD YOUR CODE HERE..]

            # 4. Map each of the connection points using the same function created above
            # [..ADD YOUR CODE HERE..]

            # 5. For each segment,
            # [..ADD YOUR CODE HERE..]

                # 6. Map the start and the end points of the segment using the same function created above
                # [..ADD YOUR CODE HERE..]

        # 7. Set each unit's centroid to the corresponding graph node's xyz attributes
        # [..ADD YOUR CODE HERE..]
        
        # 8. Return the new, mapped tessellation
        # [..ADD YOUR CODE HERE..]
        


    # *********************************
    # BEGIN: CHALLENGE 01
    # *********************************

    def pack_beams_for_stock_beams(self, origin, stock_length, safety_dist, beam_spacing, beam_offset):
        # 1. Sort beams by length in descending order (place longer beams first)
        # [..ADD YOUR CODE HERE..]
        
        # 2. Initialize visualization parameters
        stock_beams = []
        transformed_beams = []
        
        # 3. Start position for visualization
        base_x = origin.x - stock_length / 2
        base_y = origin.y + beam_spacing
        base_z = origin.z
        
        # 4. Initialize stock tracking
        stock_beams_needed = 0
        current_stocks = []  # List of current stock beams with remaining space
        
        # 2. Process each beam
        # [..ADD YOUR CODE HERE..]
            # 2.1 Get beam length and add safety distance
            # [..ADD YOUR CODE HERE..]
            
            # 2.2 Try to fit into an existing stock beam
            # [..ADD YOUR CODE HERE..]
                # 2.3 If beam fits in current stock:
                #     - Calculate position
                #     - Create position for the new beam
                #     - Transform beam.geometry
                # [..ADD YOUR CODE HERE..]
            
            # 2.4 If beam doesn't fit in any existing stock:
            #     - Create a new stock beam
            #     - Create visualization for the new stock beam
            #     - Position the beam on the new stock
            #     - Add the new stock to the list
            # [..ADD YOUR CODE HERE..]
        
        # 3. Calculate total waste
        total_waste = 0  # Replace with your calculation
        # [..ADD YOUR CODE HERE..]
        
        return stock_beams_needed, stock_beams, transformed_beams, total_waste

    # *********************************
    # END: CHALLENGE 01
    # *********************************

    # *********************************
    # BEGIN: CHALLENGE 02
    # *********************************

    def apply_attractor(self, attractor_point : Point = None) -> None:
        # 1. For each unit in the tessellation
        # [..ADD YOUR CODE HERE..]

            # 2. Calculate the attractor effect by taking the distance from the unit's centroid to the attractor point
            # [..ADD YOUR CODE HERE..]

            # 3. Calculate any additional modifications of the factor to amplify or reduce the effect
            # [..ADD YOUR CODE HERE..]

            # 4. Apply the factor to the unit's start_eccentricity or any other attribute you may have chosen
            # [..ADD YOUR CODE HERE..]

            # 5. Bonus: create a different effect if the unit is detected to be on the boundary
            # [..ADD YOUR CODE HERE..]


            # 6. Re-generate the segments of the unit calling the `.generate_segments()` method and the `.adjust_segments()` method
            # [..ADD YOUR CODE HERE..]


    # *********************************
    # END: CHALLENGE 02
    # *********************************


# 9. [Main task] Create a function that maps a point from one surface to another
# [..ADD YOUR CODE HERE..]
def map_point_to_target(point, source_surface, target_surface):
    # 9.1. Use the UV mapping technique to calculate the point on the target surface
    #      that matches the closest point on the source surface's UV space
    # [..ADD YOUR CODE HERE..]



# *********************************
# END: MAIN TASK
# *********************************

# ***************************************************************************************************
# 
# END: ASSIGNMENT 03
# 
# ***************************************************************************************************
