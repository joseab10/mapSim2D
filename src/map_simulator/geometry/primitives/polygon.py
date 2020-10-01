import numpy as np
from line import Line


class Polygon:
    """
    Polygon Class comprised of a list of ordered vertices.
    A Polygon is considered a closed region, so the first and last vertices are connected by an edge.
    No need to duplicate the first vertex in order to close it.
    """

    def __init__(self, vertices=None, compute_bounding_box=True, opacity=1.0):
        """
        Constructor.

        :param vertices: (list|ndarray) Ordered list of vertices comprising a closed polygon. Last edge connects the
                                        first and last vertices automatically, so no need to duplicate either of them.
        :param compute_bounding_box: (bool) Determines whether to compute a bounding rectangular box for this polygon.
                                            Used only because bounding boxes are themselves polygons, and must not get
                                            their own bounding box, as then they would infinitely recurse.
        :param opacity: (float: [0.0, 1.0]) Opacity level of the polygon. 0.0 means that it is totally transparent,
                                            while 1.0 is totally opaque. Used for line_intersect() to randomly determine
                                            if a line intersects it or not when opacity is set to a value other than 1.
        """

        self.vertices = None

        self.boundingBox = compute_bounding_box

        self.opacity = opacity

        self.min = None
        self.max = None

        if vertices is not None:
            self.set_vertices(vertices, compute_bounding_box)

    def set_vertices(self, vertices, compute_bounding_box=True):
        """
        Sets the vertices of a polygon after being created and recomputes its bounding box if specified.

        :param vertices: (list|ndarray) Ordered list of vertices comprising a closed polygon. Last edge connects the
                                        first and last vertices automatically, so no need to duplicate either of them.
        :param compute_bounding_box: (bool) Determines whether to compute a bounding rectangular box for this polygon.
                                            Used only because bounding boxes are themselves polygons, and must not get
                                            their own bounding box, as then they would infinitely recurse.

        :return: None
        """

        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)

        self.vertices = vertices

        if compute_bounding_box:
            self._set_bounding_box()

    def _set_bounding_box(self):
        """
        Sets the polygon's bounding box from its minimum and maximum x and y values.

        :return: (Polygon) A polygon representing a bounding box rectangle completely enclosing its parent Polygon.
        """

        x_s = self.vertices[:, 0]
        y_s = self.vertices[:, 1]
        self.min_x = np.min(x_s)
        self.min_y = np.min(y_s)
        self.max_x = np.max(x_s)
        self.max_y = np.max(y_s)

        return self.bounding_box()

    def bounding_box(self):
        """
        Gets the polygon's bounding box.

        :return: (Polygon) A polygon representing a bounding box rectangle completely enclosing its parent Polygon.
        """

        return Polygon([[self.min_x, self.min_y],
                        [self.min_x, self.max_y],
                        [self.max_x, self.max_y],
                        [self.max_x, self.max_y]], compute_bounding_box=False)

    def point_inside(self):
        """
        Check if a given point lies inside the close polygon.

        :return: (bool) True if the point lies inside the polygon, False otherwise
        """

        # TODO:
        return bool(self.vertices)

    def line_intersects(self, line):
        """
        Check if a given line segment intersects the polygon. Takes into account the polygon's opacity.

        :param line: (Line) A line segment to check whether it intersects the polygon.

        :return: (point|None) A point of the intersection closest to the line's first point p1 if the line segment
                              intersects the polygon's edges. None if it doesn't intersect.
        """

        if self.opacity < 0:
            return None

        if self.opacity < 1.0:
            reflection_prob = np.random.uniform(0.0, 1.0)
            if reflection_prob > self.opacity:
                return None

        # If polygon has more vertices than a rectangle
        if self.vertices.shape[0] > 4:
            # Check if line intersects bounding box, if not, don't even bother in checking in detail.
            if self.boundingBox:

                if not self.bounding_box().line_intersects(line):
                    return None

        min_p = None

        for i, v in enumerate(self.vertices):

            edge = Line(v, self.vertices[i - 1])

            p = line.intersects(edge)

            if p is not None:
                # Keep the point closest to the first point in the line
                if min_p is None or Line(p, line.p1).len < Line(min_p, line.p1).len:
                    min_p = p

        return min_p

    def __str__(self):
        """
        String representation of the polygon as a list of vertices for easier debugging and printing.

        :return: (string) String representation of the set of vertices.
        """

        vertex_str = "[Poly: {"

        for vertex in self.vertices:
            vertex_str += str(vertex) + ", "

        vertex_str = vertex_str[0:-2] + "}"

        vertex_str += ', Op.: '
        vertex_str += str(self.opacity)
        vertex_str += ']'

        return vertex_str

    def __repr__(self):
        """
        String representation of the polygon as a list of vertices for easier debugging and printing.

        :return: (string) String representation of the set of vertices.
        """

        return self.__str__()


if __name__ == "__main__":
    """
    Testing and Sample code
    """

    square = Polygon(np.array([[1., 1.], [1., 2.], [2., 2.], [2., 1.]]))

    ray = Line(np.array([0., 0.]), np.array([3., 2.]))
    intersection = square.line_intersects(ray)
    print("Square:", square, "intersects ray:", ray, "at", intersection)

    ray = Line(np.array([0., 0.]), np.array([1., 0]))
    intersection = square.line_intersects(ray)
    print("Square:", square, "intersects ray:", ray, "at", intersection)

    ray = Line(np.array([0., 1.]), np.array([10., 1]))
    intersection = square.line_intersects(ray)
    print("Square:", square, "intersects ray:", ray, "at", intersection)

    ray = Line(np.array([0., 0.]), np.array([10., 10.]))
    intersection = square.line_intersects(ray)
    print("Square:", square, "intersects ray:", ray, "at", intersection)