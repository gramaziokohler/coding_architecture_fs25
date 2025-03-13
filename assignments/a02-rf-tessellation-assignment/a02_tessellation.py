import math

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polygon
from compas.geometry import Vector
from compas.geometry import Frame
from compas.geometry import Transformation

from compas.tolerance import TOL
from compas.datastructures import Graph


class RFTessellation:
    # 1. Create a constructor (__init__) accepting a vertex_configuration parameter as a tuple/list of vertex counts, e.g.: [3, 6, 3, 6]
    # [..ADD YOUR CODE HERE..]
        # 1.1. Store vertex_configuration attribute and initialize a new graph instance with a default node attribute `connected_shape_edges` set to an empty list 
        # [..ADD YOUR CODE HERE..]

    # 2. Create a @property called `units` that returns the units stored in the edges of the graph
    # [..ADD YOUR CODE HERE..]


    # 3. Implement get_next_edge_index
    def get_next_edge_index(self, unit, unit_key, current_vertex):
        # Find the edge of new_unit that is connected to the current vertex, and is not already used in a connection

        # 3.1 Create a list of candidate edges
        # [..ADD YOUR CODE HERE..]

        # 3.2 Iterate over the unit's edges
        # [..ADD YOUR CODE HERE..]

            # 3.3 If edge is not already used (hint, check the graph's node_attribute `connected_shape_edges`)
            # [..ADD YOUR CODE HERE..]

                # 3.4 Prepare candidates based on the min distance of either of its ends to the current vertex
                # [..ADD YOUR CODE HERE..]

        # 3.5. Sort the candidate edges by distance and take the first if any
        # [..ADD YOUR CODE HERE..]

        # 3.6. Otherwise return None for no edge found
        # [..ADD YOUR CODE HERE..]

    # **********************************************************
    # START OF MAIN TASK WITHOUT TACKLING CHALLENGES
    # This is the simpler solution to the lookback check
    # **********************************************************
    # 4. Implement is_simple_closed_loopback
    def is_simple_closed_loopback(self, new_unit, first_unit):
        # 4.1 If the number of nodes in the graph equals the length of vertex config, we probably completed the cycle
        # [..ADD YOUR CODE HERE..]
            # 4.2 Check if any connection points between the units are close to zero in distance
            # [..ADD YOUR CODE HERE..]
            # 4.3 If so, add the edge and node attributes `connected_shape_edges` to the graph
            # [..ADD YOUR CODE HERE..]

        # 4.4 Otherwise return false        

    # MAIN TASK: Implement apply_single_rule_cycle (without tackling the multi rule challenge 02)
    def apply_single_rule_cycle(self, rule, vertex_index=0):
        # 5.1 Check if the first unit (unit1) is already in the graph, if not, add it as a node
        # [..ADD YOUR CODE HERE..]

        # 5.2 Get the vertex coordinates of the point at the given vertex_index (hint, use unit1's shape)
        # [..ADD YOUR CODE HERE..]
        
        # 5.3 Create a for..loop iteration with a max of 20 iterations, just to make sure you don't accidentally freeze Grasshopper
        # [..ADD YOUR CODE HERE..]

            # 5.4 Get the next edge index (hint, use get_next_edge_index)
            # [..ADD YOUR CODE HERE..]

            # 5.5 Apply the rule to get the new unit
            # [..ADD YOUR CODE HERE..]

            # 5.6 Add the new unit as a node to the graph
            # [..ADD YOUR CODE HERE..]

            # 5.7 Append next_edge to `connected_shape_edges` node attribute 
            # [..ADD YOUR CODE HERE..]

            # 5.7 Add the edge to the graph connecting unit1 and new unit
            # [..ADD YOUR CODE HERE..]

            # 5.8 Check if we have a closed loopback, if so, return
            # [..ADD YOUR CODE HERE..]

            # 5.9 Reverse the rule to prepare it for next iteration (you'll need to assign the new_unit to the next first unit)
            # [..ADD YOUR CODE HERE..]


    # **********************************************************
    # END OF MAIN TASK WITHOUT TACKLING CHALLENGES
    # **********************************************************

    # **********************************************************
    # START OF MAIN TASK WITH BOTH CHALLENGES
    # This is the more complete solution to the exercise
    # **********************************************************

    # 6. Implement a loopback check that takes into account all existing units
    def is_closed_loopback(self, new_unit):
        # 6.1 Get the connected_shape_edges of the new unit from the graph
        # [..ADD YOUR CODE HERE..]

        # 6.2 For each edge in the newly added unit...
        # [..ADD YOUR CODE HERE..]
            # 6.3. If the edge is already connected (i.e. in the list retrived above), ignore it and continue
            # [..ADD YOUR CODE HERE..]

            # 6.4 If it's not yet connected, get the corresnpoding connection point of the edge to check if it forms a loopback
            # [..ADD YOUR CODE HERE..]

            # 6.5 First collect all connection points from all other units that are NOT equal to new_unit
            # [..ADD YOUR CODE HERE..]
            # 6.6 In the list, add the distance between the connection point on the new unit, and each other connection point
            # [..ADD YOUR CODE HERE..]
            
            # 6.7 Sort connection points by distance to p1
            # [..ADD YOUR CODE HERE..]
            
            # 6.8 Take the first/closest point (since it's sorted by distance), and check if the distance is equals to zero
            # [..ADD YOUR CODE HERE..]
            
            # 6.9 If so, Add the edge between the two units, as well as appending to the connected_shape_edges node attributes to the graph
            # [..ADD YOUR CODE HERE..]

            # 6.10 Return True, we found a loopback
            # [..ADD YOUR CODE HERE..]

        # 6.11 If nothing is found, return False
        # [..ADD YOUR CODE HERE..]


    # 7. Create a vertex pair iterator
    def create_vertex_pair_iterator(self):
        # [..ADD YOUR CODE HERE..]

    # 8. Implement a get next vertex pair from the iterator
    def get_next_vertex_pair(self):
        # [..ADD YOUR CODE HERE..]


    # MAIN TASK: Implement apply_rules_cycle including the challenge to handle more than 1 rule for application!
    def apply_rules_cycle(self, rules, vertex_index=0):
        # 9.1 Check if the first unit of the first rule is already in the graph, if not, add it as a node
        # [..ADD YOUR CODE HERE..]

        # 9.2 Get the vertex coordinates of the point at the given vertex_index (hint, use the unit's shape)
        # [..ADD YOUR CODE HERE..]

        # 9.3 Create an empty dictionary of available rule
        # [..ADD YOUR CODE HERE..]        
        available_rules = {}

        # 9.4 For each rule, add it to the dictionary using the len of edges of unit 1 and 2 and key, e.g. `(3, 6)` for a rule connecting a triangle to a hexagon
        # [..ADD YOUR CODE HERE..]
        # 9.5 If the lens of edges are different, i.e, if it's not a (3, 3), or (4, 4), it's useful to add a second rule with the reversed rule and reversed key
        # [..ADD YOUR CODE HERE..]

        # 9.6 Get the next vertex pair from the iterator
        # [..ADD YOUR CODE HERE..]
        
        # 9.7 Select the rule from available rules dict, using the next vertex pair``
        # [..ADD YOUR CODE HERE..]

        # 9.8 Create a for..loop iteration with a max of 20 iterations, just to make sure you don't accidentally freeze Grasshopper
        # [..ADD YOUR CODE HERE..]
            # 9.9 Get the next edge index (hint, use get_next_edge_index)
            # [..ADD YOUR CODE HERE..]

            # 9.10 Apply the rule to get the new unit
            # [..ADD YOUR CODE HERE..]

            # 9.11 Add the new unit as a node to the graph
            # [..ADD YOUR CODE HERE..]

            # 9.12 Append next_edge to `connected_shape_edges` node attribute 
            # [..ADD YOUR CODE HERE..]

            # 9.13 Add the edge to the graph connecting unit1 and new unit
            # [..ADD YOUR CODE HERE..]

            # 9.14 Check if we have a closed loopback, if so, return
            # [..ADD YOUR CODE HERE..]
            
            # 9.15 Get the next vertex pair and prepare the rule for the next iteration:
            #      (if the next pair is identical to the current, just reverse the rule, otherwise, get a new from the dictionary)
            # [..ADD YOUR CODE HERE..]


    # **********************************************************
    # END OF MAIN TASK WITH BOTH CHALLENGES
    # **********************************************************

# -------------------------------------------------------------------------------------------------
# BELOW THIS LINE, YOU CAN MOSTLY USE YOUR CODE FROM PREVIOUS EXERCISES of RFUnit and RFGrammarRule
# -------------------------------------------------------------------------------------------------

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
        # NEW: This is a new attribute that we have added for this assignment to ease keeping track of units!!
        self.key = None

    @property
    def centroid(self):
        return self.shape.centroid

    @property
    def edges(self):
        return self.shape.lines

    def generate_segments(self):
        self.connection_points = []
        self.segments = []

        # For each edge
        for edge in self.edges:
            # Calculate midpoint of the edge
            midpoint = edge.midpoint

            # Calculate eccentricities
            start = self.centroid + edge.vector * self.start_eccentricity
            end = midpoint + edge.vector * self.end_eccentricity

            # Create an instance of the RodSegment and append to the correct list
            segment = RodSegment(start, end)
            self.segments.append(segment)

            # Create an instance of ConnectionPoint
            # Store the connection point
            _point, param = edge.closest_point(end, return_parameter=True)
            vector = Vector.from_start_end(start, end).unitized()
            connection = ConnectionPoint(end, param, vector)
            self.connection_points.append(connection)


    def adjust_segments(self):
        pass

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
    def __init__(self, unit1, unit2, edge1, edge2, mirror):
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


    # NEW: For convenience, add a `reversed()` method that returns the reversed rule!
    # [..ADD YOUR CODE HERE..]
