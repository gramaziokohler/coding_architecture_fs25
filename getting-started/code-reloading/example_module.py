from compas.geometry import Polygon

class House:
    def __init__(self, name, area):
        self.name = name
        self.area = area
        self.rectangle = None

    def generate_rectangle(self):
        side_length = (self.area ** 0.5) # square root
        self.rectangle = Polygon.from_sides_and_radius_xy(4, side_length / 2)


    def __str__(self):
        return f"House(name={self.name}, area={self.area}, rectangle={self.rectangle})"
    
