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

        # ADDITIONAL TASK: Apply attractor point/curve effect
        # Apply local changes to the units based on an attractor point/curve

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
            
            # # assign key to beam
            # beam.key = 

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
        vectors = []
        main_beams = []
        

        # For each unit, gather beams
        for unit in self.units:
            beams = []
            
            for seg in unit.segments:
                if seg.rod and seg.rod.beam:
                    beams.append(seg.rod.beam)
                    # vectors.append(seg.rod.beam.centerline.vector)
            # print(beams)
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

                if int_a is None or int_b is None:
                    print(f"No intersection found between beams {i} and {j}")
                    continue

                point_a = Point(*int_a)
                point_b = Point(*int_b)

                intersection_points1.append(point_a)
                intersection_points2.append(point_b)

                print(f"Intersection Points: {point_a, point_b}")

                # for mapped tessellation
                if point_a is not None and point_b is not None:
                    
                    print(f"Intersection found between beams {i} and {j}")

                    # Ensure correct ordering of main & cross beams based on Z comparison
                    if point_a.z > point_b.z:
                        main_beam, cross_beam = beam_a, beam_b
                    else:
                        main_beam, cross_beam = beam_b, beam_a

                    # print(f"Main Beam in case of Intersection: {main_beam}, Cross Beam: {cross_beam}")
                    

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
                    
                    # Create joint
                    XLapJoint.create(model, main_beam, cross_beam, flip_side, cut_plane_bias=cut_plane_bias)

        return intersection_points1, intersection_points2, plane_cut_vectors, offset_vectors

    def find_optimal_engraving_spot(self, model):
        # Make sure we're working with a valid model
        if not model or not hasattr(model, 'elements'):
            raise ValueError("A valid TimberModel must be provided")

        print("--------------------------")
        print("starting now")
        print("--------------------------")
        # Dictionary to store intersection parameters for each beam
        beam_intersection_params = {}
        beam_objects = {}
        
        # For each unit, gather beams and find intersections
        for unit in self.units:
            beams = []
            
            # Collect all beams in the unit
            for seg in unit.segments:
                if seg.rod and seg.rod.beam:
                    beam = seg.rod.beam
                    beams.append(beam)
                    
                    # Use beam object as key instead of guid
                    beam_id = id(beam)  # Use object id as unique identifier
                    if beam_id not in beam_intersection_params:
                        beam_intersection_params[beam_id] = []
                        beam_objects[beam_id] = beam
            
            num_beams = len(beams)
            if num_beams < 2:
                continue
            
            # Find intersections between consecutive beams (wrapping around)
            for i in range(num_beams):
                j = (i + 1) % num_beams
                beam_a = beams[i]
                beam_b = beams[j]
                
                # Find intersection between centerlines
                result = intersection_line_line(beam_a.centerline, beam_b.centerline, tol=0.4)
                
                if result and result[0] is not None and result[1] is not None:
                    int_point_a, int_point_b = result
                    
                    # Calculate parameter along each centerline (0.0 at start, 1.0 at end)
                    _, param_a = beam_a.centerline.closest_point(Point(*int_point_a), return_parameter=True)
                    _, param_b = beam_b.centerline.closest_point(Point(*int_point_b), return_parameter=True)
                    
                    # Store the parameters using object id
                    beam_a_id = id(beam_a)
                    beam_b_id = id(beam_b)
                    beam_intersection_params[beam_a_id].append(param_a)
                    beam_intersection_params[beam_b_id].append(param_b)
        
        # For each beam, find the largest gap between intersection parameters and store as attribute
        for beam_id, params in beam_intersection_params.items():
            beam = beam_objects[beam_id]
            
            # Sort parameters to find gaps
            if not params:
                # If no intersections, use the middle of the beam
                middle_param = 0.5
                middle_point = beam.centerline.point_at(middle_param)
                # Store directly as beam attribute
                beam.attributes["engraving_spot"] = (middle_param, middle_point)
                continue
                
            # Add start and end parameters (0.0 and 1.0) to consider gaps at the ends
            all_params = [0.0] + sorted(params) + [1.0]
            
            # Find the largest gap
            largest_gap = 0.0
            largest_gap_start = 0.0
            
            for i in range(len(all_params) - 1):
                gap = all_params[i+1] - all_params[i]
                if gap > largest_gap:
                    largest_gap = gap
                    largest_gap_start = all_params[i]
            
            # Calculate the middle of the largest gap
            middle_param = largest_gap_start + (largest_gap / 2.0)
            
            # Get the point at this parameter
            middle_point = beam.centerline.point_at(middle_param)
            
            # Store directly as beam attribute
            beam.attributes["engraving_spot"] = (middle_param, middle_point)

            print(f"Engraving Spot for {beam.key}: {middle_param}, {middle_point}")
        
        # Collect all beams that have engraving spots
        engraved_beams = [beam for beam in beam_objects.values() if "engraving_spot" in beam.attributes]
        print(f"Found {len(engraved_beams)} beams with engraving spots")
        
        # Return all beams with engraving spots for further processing
        return engraved_beams

    def map_to_surface(self, source_brep, target_brep, tolerance=0.015):
        if len(source_brep.faces) > 1:
            raise Exception("Source brep must have a single face.")
        if len(target_brep.faces) > 1:
            raise Exception("Target brep must have a single face.")

        source_face =  source_brep.faces[0]
        target_face =  target_brep.faces[0]
        source_surface =  source_face.nurbssurface
        target_surface =  target_face.nurbssurface

        tesselation = self   #.copy()

        to_delete = []
        for unit in tesselation.units:
            # Make units outside the source brep connect all beams at the centroid
            # (basically removing start eccentricity without regenerating segments)
            if not is_unit_on_brepface(source_face, unit, 0.015):
                for segment in unit.segments:
                    segment.start = Point(*unit.centroid)

                to_delete.append(unit.key)
                continue

            new_points = [map_point_to_target(p, source_surface, target_surface) for p in unit.shape.points]
            if len(new_points) == 0:
                print(f"Unit {unit.key} has no points after mapping")
                continue
            unit.shape.points = new_points

            for connection_point in unit.connection_points:
                connection_point.point = map_point_to_target(connection_point.point, source_surface, target_surface)

            for segment in unit.segments:
                segment.start = map_point_to_target(segment.start, source_surface, target_surface)
                segment.end = map_point_to_target(segment.end, source_surface, target_surface)

                mapped_start = map_point_to_target_brep(segment.start, source_brep, target_brep)
                mapped_end = map_point_to_target_brep(segment.end, source_brep, target_brep)

                start_inside = is_point_on_brepface(target_face, mapped_start, tolerance)
                end_inside = is_point_on_brepface(target_face, mapped_end, tolerance)

                if not start_inside and not end_inside:
                    print("Found segment fully outside the boundary, leave them out")
                    continue

                segment.start = mapped_start
                segment.end = mapped_end


        for rod in self.rods:
            if rod.segment2 is not None:
                start_inside = is_point_on_brepface(target_face, rod.line.start, tolerance)
                end_inside = is_point_on_brepface(target_face, rod.line.end, tolerance)

                # Both sides are outside, we need to remove the link to this rod from its segments
                if not start_inside and not end_inside:
                    rod.segment1.rod = None
                    rod.segment2.rod = None

                # One of the two sides of the rod is outside, we need to remove the second segment
                if not (start_inside and end_inside):
                    rod.segment2 = None

        # Re-assign xyz to graph nodes
        for unit in tesselation.units:
            tesselation.graph.node_attributes(unit.key, "xyz", list(unit.centroid))

        for key in to_delete:
            tesselation.graph.delete_node(key)
        
        return tesselation


def map_point_to_target_brep(point, source_brep, target_brep):
    _closest, (u, v) = source_brep.faces[0].nurbssurface.closest_point(point, return_parameters=True)
    pt = target_brep.faces[0].native_face.PointAt(u, v)
    return Point(pt.X, pt.Y, pt.Z)


def map_point_to_target(point, source_surface, target_surface):
    _, uv_params = source_surface.closest_point(point, return_parameters=True)
    return target_surface.point_at(*uv_params)


def is_point_on_brepface(brepface, point, tolerance):
    closest, (u, v) = brepface.nurbssurface.closest_point(point, return_parameters=True)
    return closest.distance_to_point(point) < tolerance and int(brepface.native_face.IsPointOnFace(u, v)) == 1

def is_unit_on_brepface(source_face, unit, tolerance):
    return any([is_point_on_brepface(source_face, point, tolerance) for point in unit.shape.points])

# -----------------------------------------------------------------
# Pack Beams for Stock Beams
# -----------------------------------------------------------------

from compas.geometry import Line, Point, Transformation, Translation, Vector, Frame
from compas_timber.elements import Beam


def pack_beams_for_stock_beams(beams, origin, stock_length, safety_dist, beam_spacing, beam_offset):
    # Sort beams by length in descending order (place longer beams first)
    # First extract the lengths
    beam_lengths = [(i, beam.centerline.length) for i, beam in enumerate(beams)]
    # Sort by length
    sorted_indices = [i for i, _ in sorted(beam_lengths, key=lambda x: x[1], reverse=True)]
    # Create new sorted list
    sorted_beams = [beams[i] for i in sorted_indices]
    
    # Initialize visualization parameters
    stock_beams = []
    transformed_beams = []
    
    # Start position for visualization
    base_x = origin.x - stock_length / 2
    base_y = origin.y + beam_spacing
    base_z = origin.z
    
    # Initialize stock tracking
    stock_beams_needed = 0
    current_stocks = []  # List of current stock beams with remaining space
    
    # Process each beam
    for beam in sorted_beams:
        beam_length = beam.centerline.length
        beam_length_tol = beam_length + safety_dist  # Add safety distance
        
        fitted = False
        
        # Try to fit into an existing stock beam
        for stock_idx, stock in enumerate(current_stocks):
            if stock["remaining"] >= beam_length_tol:
                # This beam fits in the current stock
                stock_y_position = base_y + (stock_idx * beam_spacing)
                cut_beam_y_position = stock_y_position + beam_offset
                
                # Create position for the new beam
                start = Point(stock["next_pos"], cut_beam_y_position, base_z)
                end = Point(stock["next_pos"] + beam_length, cut_beam_y_position, base_z)

                initial_frame = beam.frame
                new_frame = Frame(start, Vector(1, 0, 0), Vector(0, 0, 1))
                transformation = Transformation.from_frame_to_frame(initial_frame, new_frame)

                # Transform the beam.geometry
                new_beam = beam.geometry.transformed(transformation)

                transformed_beams.append(new_beam)
                
                # Update stock information
                stock["remaining"] -= beam_length_tol
                stock["next_pos"] += beam_length_tol
                fitted = True
                break
        
        if not fitted:
            # Create a new stock beam
            stock_beams_needed += 1
            stock_idx = len(current_stocks)
            
            # Create visualization for the new stock beam
            stock_y_position = base_y + (stock_idx * beam_spacing)
            cut_beam_y_position = stock_y_position + beam_offset
            
            # Create stock beam
            stock_start = Point(base_x, stock_y_position, base_z)
            stock_end = Point(base_x + stock_length, stock_y_position, base_z)
            stock_centerline = Line(stock_start, stock_end)
            stock_beam = Beam.from_centerline(stock_centerline, beam.width, beam.height, Vector(0, 1, 0))
            stock_beams.append(stock_beam)
            
            # Create a new beam at the target position
            start = Point(base_x, cut_beam_y_position, base_z)
            end = Point(base_x + beam_length, cut_beam_y_position, base_z)
            
            initial_frame = beam.frame
            new_frame = Frame(start, Vector(1, 0, 0), Vector(0, 0, 1))
            transformation = Transformation.from_frame_to_frame(initial_frame, new_frame)
            
            # Transform the beam.geometry
            new_beam = beam.geometry.transformed(transformation)

            transformed_beams.append(new_beam)
            
            
            # Add the new stock to the list
            current_stocks.append({
                "remaining": stock_length - beam_length_tol,
                "next_pos": base_x + beam_length_tol
            })
    
    # Calculate total waste
    total_waste = sum(stock["remaining"] for stock in current_stocks)
    
    return stock_beams_needed, stock_beams, transformed_beams, total_waste


def generate_beam_stats(beams, stock_beams_needed, stock_length, total_waste, lm_price, transport_cost, wood_density):
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


def reconstruct_tessellation(tessellation, rod_lines):
    # this function updates an existing tessellation by replacing the existing rod lines with the new ones
    # replacement is by index

    # create new tessellation
    tessellation = tessellation.copy()
    
    # replace rod lines
    rods = tessellation.rods

    rod_lines = line_to_compas(rod_lines)
    
    for i, rod_line in enumerate(rod_lines):
        if i < len(rods):
            rod = rods[i]
            
            # Update the rod segments based on the new rod line
            if rod.segment2 is not None:
                # For rods with two segments
                # Update the first segment
                rod.segment1.start = rod_line.start
                rod.segment1.end = rod_line.midpoint
                
                # Update the second segment
                rod.segment2.start = rod_line.end
                rod.segment2.end = rod_line.midpoint
            else:
                # For rods with a single segment
                rod.segment1.start = rod_line.start
                rod.segment1.end = rod_line.end
    
    return tessellation


def add_beams_to_ctmodel(model, custom_lines):
    """Add new beams to an existing compas timber model and create XLap joints at intersections.
    
    Args:
        model: The existing TimberModel to add beams to
        custom_lines: List of lines to create beams from
        
    Returns:
        list: List of beam geometries for visualization
    """
    # r: compas, compas_timber
    # venv: ca-fs25

    from compas_rhino import DevTools
    DevTools.ensure_path()

    from compas.geometry import Line, Point, Polygon, Vector, Frame
    from compas.geometry import Transformation, Translation
    from compas.geometry import intersection_line_line
    from compas_rhino.conversions import line_to_compas

    from compas.tolerance import TOL
    from compas.datastructures import Graph

    from compas_timber.connections import XLapJoint
    from compas_timber.elements import Beam
    from compas_timber.model import TimberModel

    # Create a copy of the model to work with
    model_temp = model.copy()
    z_vec = Vector(0, 0, 1)
    
    # List to store newly created beams
    new_beams = []
    
    # Add new beams from custom lines
    for line in custom_lines:
        # Convert to compas line if needed
        line = line_to_compas(line)
        
        # Create new beam
        new_beam = Beam.from_centerline(line, 0.06, 0.08, z_vec)
        model_temp.add_element(new_beam)
        new_beams.append(new_beam)
        
        # Find intersections with existing beams (excluding the newly added beam)
        for beam in model_temp.beams:
            # Skip if comparing with itself
            if beam == new_beam:
                continue
                
            # Get centerlines
            line1 = new_beam.centerline
            line2 = beam.centerline
            
            # Check for intersection
            result = intersection_line_line(line1, line2, tol=0.3)
            
            # If intersection found, create joint
            if result and None not in result:
                int_point_a, int_point_b = result
                
                # Only create joint if the intersection points are close enough
                if int_point_a.distance_to_point(int_point_b) < 0.01:  # Small tolerance
                    # Create XLap joint between the beams
                    # The flip parameter alternates to create varied joint orientations
                    # The cut_plane_bias parameter controls the position of the joint along the beam
                    try:
                        XLapJoint.create(model_temp, new_beam, beam, flip=False, cut_plane_bias=0.5)
                        print(f"Created XLap joint between beams at {int_point_a}")
                    except Exception as e:
                        print(f"Failed to create joint: {e}")
    
    # Process all joinery in the model
    model_temp.process_joinery()
    
    # Collect geometries for output
    geometry = []
    for beam in model_temp.beams:
        geometry.append(beam.geometry)
    
    return geometry

    

