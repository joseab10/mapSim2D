import numpy as np


class Line:
    """
    Line segment class comprised of two endpoints
    """

    def __init__(self, p1=None, p2=None):
        """
        Constructor

        :param p1: First point of the Line segment.
        :param p2: Second point of the Line segment.
        """

        self.p1 = p1
        self.p2 = p2

        self.len = self._length()
        self.slope = self._slope()

    def _slope(self):
        """
        Returns the slope of the line segment

        :return: (float) Slope of the line segment
        """

        diff = self.p2 - self.p1
        return np.arctan2(diff[1], diff[0])

    def _length(self):
        """
        Returns the length of the line segment

        :return: (float) Length of the line segment
        """

        diff = self.p2 - self.p1
        return np.sqrt(np.dot(diff, diff))

    def intersects(self, line2, outside_segments=False):
        """
        Detects whether this line intersects another line segment.
        Algorithm from:
            https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect#answer-565282

        :param line2: (Line) Second line to determine the intersection with
        :param outside_segments: (bool) Determines whether the intersection needs to fall within the two line segments
                                 in order to count. If True, then the lines are considered of infinite length.

        :return: (ndarray) The point where the two lines intersect each other. If collinear, then the point closest
                            to self.p1. Returns None if the lines don't intersect.
        """

        p = self.p1
        q = line2.p1
        r = self.p2 - p
        s = line2.p2 - q

        denominator = np.cross(r, s)
        q_p = q - p
        q_pxr = np.cross(q_p, r)
        r_len = np.dot(r, r)

        # Parallel
        if denominator == 0:
            # Collinear
            if q_pxr == 0:
                t0 = np.dot(q_p, r) / r_len
                t1 = t0 + np.dot(s, r) / r_len

                sr = np.dot(s, r)
                if sr < 0:
                    tmp = t0
                    t0 = t1
                    t1 = tmp

                if (0 <= t1 and 1 >= t0) or outside_segments:
                    return p + t0 * r

            # else:
            # Parallel and not intersecting

        # Not Parallel
        else:
            t = np.cross(q_p, s) / denominator
            u = q_pxr / denominator

            if (0 <= t <= 1 and 0 <= u <= 1) or outside_segments:
                return p + t * r

        # Lines do not intersect
        return None

    def is_parallel(self, line2):
        """
        Determines whether this line is parallel to another line

        :param line2: (Line) Second line to check parallelism with

        :return: (bool) True if lines are parallel, False otherwise
        """
        return self.slope == line2.slope

    def __str__(self):
        """
        String representation

        :return: A serialized string for more convenient printing and debugging
        """

        return "[Line: {" + str(self.p1) + ", " + str(self.p2) + "}]"

    def __repr__(self):
        """
        String representation

        :return: A serialized string for more convenient printing and debugging
        """

        return self.__str__()


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


def rotate2d(theta):
    """
    Compute a 2D rotation matrix given an angle.

    :param theta: (float) Angle to rotate about the z axis.

    :return: (ndarray) A 2x2 rotation matrix.
    """

    s, c = np.sin(theta), np.cos(theta)

    return np.array([[c, -s], [s, c]]).reshape((2, 2))


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
