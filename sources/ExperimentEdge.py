from sources.Edge import Edge


class ExperimentEdge(Edge):
    def __init__(self, u: int, v: int, edge_id: int, weight: complex, u_colour, v_colour):
        super().__init__(u, v, edge_id)
        self.weight = weight
        self.u_colour = u_colour
        self.v_colour = v_colour

    def get_weight(self):
        return self.weight

    def get_colour(self, vertex):
        """
        Returns the colour of the vertex.
        :param vertex: int
        :return: str
        """
        if vertex == self.u:
            return self.u_colour
        return self.v_colour

    def weight_to_string(self):
        """
        Returns the weight as a string.
        :return: str
        """
        # First, we get rid of the unnecessary trailing zeros
        real_part = self.weight.real
        imaginary_part = self.weight.imag
        if real_part == int(real_part):
            real_part = int(real_part)
        if imaginary_part == int(imaginary_part):
            imaginary_part = int(imaginary_part)

        if imaginary_part == 0:  # Pure real number
            return str(real_part)
        if real_part == 0:  # Pure imaginary number
            if imaginary_part == 1:
                return "i"
            if imaginary_part == -1:
                return "-i"
            return str(imaginary_part) + "i"
        # Else, we have a complex number
        if imaginary_part > 0:  # The imaginary part is positive
            if imaginary_part == 1:
                return str(real_part) + "+i"
            return str(real_part) + "+" + str(imaginary_part) + "i"
        # Else, the imaginary part is negative
        if imaginary_part == -1:
            return str(real_part) + "-i"
        return str(real_part) + "-" + str(-imaginary_part) + "i"

    def to_dict(self):
        return {
            "u": self.u,
            "v": self.v,
            "edge_id": self.id,
            "weight": {
                "real": self.weight.real,
                "imaginary": self.weight.imag
            },
            "u_colour": self.u_colour,
            "v_colour": self.v_colour
        }

    def __copy__(self):
        return ExperimentEdge(self.u, self.v, self.id, self.weight, self.u_colour, self.v_colour)

    def __str__(self):
        return "{" + f"{self.u}({self.u_colour})--{self.weight}--{self.v}({self.v_colour})" + "}"

    def __repr__(self):
        return str(self)
