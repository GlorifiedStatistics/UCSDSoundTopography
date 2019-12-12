import numpy as np


class TopographicalLineMap:
    def __init__(self, base_dim):
        """
        A topographical sound map that is 2-dimensional
        :param base_dim: (length, width, height) of base of map. Leave height = 0 if you don't want a rectangular prism
            as the base of this map.
        """
        self.base_dim = base_dim
        self.points = np.array([])

    def add_points(self, points, shift_points=False):
        """
        Add the given list of points to the map.
        :param points: the points to add. Should be an array of shape (n, 2) for n points to add. Each point should
            be an x-y coordinate pair
        :param shift_points: if True, shifts all points upwards until there are no negative values instead of raising
            an error
        :raises ValueError: if any of the x-coordinates are negative, if any of the x-coordinates go further than the
            length of this map, or if any of the y-coordinates are negative
        :raises TypeError: if points is not a 2d list of points with shape (n, 2)
        """
        if not isinstance(points, np.ndarray):
            self.add_points(np.array(points))
            return

        # Check for points outside the range, or bad dimensions
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise TypeError("Points should be a list of 2d coordinates of shape (n, 2). Shape is: %s" % points.shape)

        if len(points[points[:, 0] < 0]) != 0:
            raise ValueError("No x-coordinates can be negative")

        if len(points[points[:, 0] > self.base_dim[0]]) != 0:
            raise ValueError("Some x-coordinates go outside the range")

        if len(points[points[:, 1] < 0]) != 0:
            if shift_points:
                points[:, 1] -= np.min(points[:, 1])
            else:
                raise ValueError("Some y-coordinates are negative")

        self.points = list(self.points)
        for p in points:
            self.points.append(p)
        self.points = np.array(self.points)

    def get_triangles(self, normalize_height=None, const_mult=None, const_shift=None, print_points=False):
        """
        Build the triangles to make this surface. Returns a list of tuples. Each tuples contains three vertices,
            where each vertex is a tuple containing the x-y-z coordinates for that vertex. The vertices are listed in
            the order that makes the normal vector point outward (based on the right-hand rule)
        :param print_points: whether or not to print the points to the screen before computing triangles
        :param normalize_height: the height the maximum value should reach to. Overrides const_mult, const_shift
        :param const_mult: a value to multiply all y-coordinates by before computing triangles
        :param const_shift: a value to shift all y-coordinates by after const_mult
        :return: a list of tuples of tuples describing all of the triangles that make this topographical map
        """
        l, w, h = self.base_dim

        # Build the base first
        tris = [
            # The front face
            ((0.0, 0.0, 0.0), (l, h, 0.0), (0.0, h, 0.0)),
            ((0.0, 0.0, 0.0), (l, 0.0, 0.0), (l, h, 0.0)),

            # The right face
            ((l, 0.0, 0.0), (l, h, w), (l, h, 0.0)),
            ((l, 0.0, 0.0), (l, 0.0, w), (l, h, w)),

            # The back face
            ((l, 0.0, w), (0.0, h, w), (l, h, w)),
            ((l, 0.0, w), (0.0, 0.0, w), (0.0, h, w)),

            # The left face
            ((0.0, 0.0, 0.0), (0.0, h, 0.0), (0.0, h, w)),
            ((0.0, 0.0, 0.0), (0.0, h, w), (0.0, 0.0, w)),

            # The bottom face
            ((0.0, 0.0, 0.0), (0.0, 0.0, w), (l, 0.0, w)),
            ((0.0, 0.0, 0.0), (l, 0.0, w), (l, 0.0, 0.0)),
        ]

        # Build the topological triangles

        # Remove the points with x-coord = 0 since those triangles will be built manually at the end. Sort over rows
        points = self.points[self.points[:, 0] != 0]
        points = points[points[:, 0].argsort()]

        if const_mult:
            points[:, 1] *= const_mult

        if const_shift:
            points[:, 1] += const_shift

        # Make the max height normalize_height if needed
        if normalize_height:
            max_h = np.max(points[:, 1])
            points[:, 1] = points[:, 1] * (float(normalize_height) / max_h)

        if print_points:
            print(points)

        # Build the points for the values
        prev = 0.0
        for x, y in points:
            tris.append(((prev, h, 0.0), (x, h, 0.0), (x, y + h, w / 2.0)))
            tris.append(((prev, h, w), (x, y + h, w / 2.0), (x, h, w)))
            prev = x

        # Build triangles that connect points together
        for i in range(len(points) - 1):
            p1x, p1y = points[i]
            p2x, p2y = points[i + 1]
            tris.append(((p1x, h, 0.0), (p2x, p2y + h, w / 2.0), (p1x, p1y + h, w / 2.0)))
            tris.append(((p1x, h, w), (p1x, p1y + h, w / 2.0), (p2x, p2y + h, w / 2.0)))

        # Build last two triangles for the last point
        tris.append(((points[-1][0], h, 0.0), (l, h, 0.0), (points[-1][0], points[-1][1] + h, w / 2.0)))
        tris.append(((points[-1][0], h, w), (points[-1][0], points[-1][1] + h, w / 2.0), (l, h, w)))

        # Build the edges
        tris.append(((0.0, h, w), (0.0, h, 0.0), (points[0][0], points[0][1] + h, w / 2.0)))
        tris.append(((l, h, 0.0), (l, h, w), (points[-1][0], points[-1][1] + h, w / 2.0)))

        return tris


class MultiTopographicalLineMap:
    def __init__(self, base_dim):
        """
        A topographical sound map that is 3-dimensional. Does multiple different line maps and smashes them together.
        :param base_dim: (length, width, height) of base of a single map. Leave height = 0 if you don't want a
            rectangular prism as the base of this map.
        """
        self.base_dim = base_dim
        self.points = None

    def add_points(self, points, shift_points=None):
        """
        Adds a 2-d array of values to this map. Points should be a 2-d array where each value corresponds to a height,
            each row a single measurement type, and each column a segment of that measurement.
            Assumes all points are evenly spread out along line.
            For example: if you wanted to measure the amount of each note, the values would be the amount of that note,
                the rows would be each note (A, A#, B, ...), and the columns would be each time slice.
        :param points: a 2-d array of points
        """
        if len(points.shape) != 2:
            raise TypeError("points must be a 2-d array, instead has shape: %s" % (points.shape, ))
        if points.shape[0] < 2:
            raise TypeError("points must have at least 2 rows, this is a multi-line map. Instead, shape is: %s"
                            % points.shape)
        if np.min(points) < 0:
            if shift_points:
                points += np.min(points)
            else:
                raise ValueError("points has negative values and shift_points is not true")
        self.points = points

    def get_triangles(self, normalize_height=None, const_mult=None, const_shift=None):
        """
        Build the triangles to make this surface. Returns a list of tuples. Each tuples contains three vertices,
            where each vertex is a tuple containing the x-y-z coordinates for that vertex. The vertices are listed in
            the order that makes the normal vector point outward (based on the right-hand rule).
        :param normalize_height: the height the maximum value should reach to. Overrides const_mult, const_shift
        :param const_mult: a value to multiply all y-coordinates by before computing triangles
        :param const_shift: a value to shift all y-coordinates by after const_mult
        :return: a list of tuples of tuples describing all of the triangles that make this topographical map
        """
        # length, single_width, height of base
        l, s_w, h = self.base_dim
        points = self.points
        num_rows = self.points.shape[0]
        x_step = l / float(self.points.shape[1])
        w = s_w * num_rows  # Make the width cover all of the rows

        # Multiply by const_mult
        if const_mult is not None:
            points *= const_mult

        # Shift up if need be
        if const_shift is not None:
            points += const_shift

        # Normalize height if need be
        if normalize_height is not None:
            max_h = np.max(points)
            points *= float(normalize_height) / max_h




        # Build the base first
        tris = [
            # The front face
            ((0.0, 0.0, 0.0), (l, h, 0.0), (0.0, h, 0.0)),
            ((0.0, 0.0, 0.0), (l, 0.0, 0.0), (l, h, 0.0)),

            # The right face
            ((l, 0.0, 0.0), (l, h, w), (l, h, 0.0)),
            ((l, 0.0, 0.0), (l, 0.0, w), (l, h, w)),

            # The back face
            ((l, 0.0, w), (0.0, h, w), (l, h, w)),
            ((l, 0.0, w), (0.0, 0.0, w), (0.0, h, w)),

            # The left face
            ((0.0, 0.0, 0.0), (0.0, h, 0.0), (0.0, h, w)),
            ((0.0, 0.0, 0.0), (0.0, h, w), (0.0, 0.0, w)),

            # The bottom face
            ((0.0, 0.0, 0.0), (0.0, 0.0, w), (l, 0.0, w)),
            ((0.0, 0.0, 0.0), (l, 0.0, w), (l, 0.0, 0.0)),
        ]

        # Build left wall

        # Go through each row building bottom half triangles
        for i in range(points.shape[0]):
            tris.append(((0.0, h, i * s_w),
                         (x_step / 2.0, points[-i - 1][0] + h, i * s_w + s_w / 2.0),
                         (0.0, h, (i + 1.0) * s_w)))
        # Go through each row building top half triangles
        for i in range(points.shape[0] - 1):
            tris.append(((0, h, (i + 1) * s_w),
                         (x_step / 2.0, points[-i - 1][0] + h, i * s_w + s_w / 2.0),
                         (x_step / 2.0, points[-i - 2][0] + h, (i + 1.0) * s_w + s_w / 2.0)))

        # Build right wall

        # Go through each row building bottom half triangles
        for i in range(points.shape[0]):
            tris.append(((l, h, i * s_w),
                         (l - x_step / 2.0, points[-i - 1][-1] + h, i * s_w + s_w / 2.0),
                         (l, h, (i + 1.0) * s_w)))
        # Go through each row building top half triangles
        for i in range(points.shape[0] - 1):
            tris.append(((l, h, (i + 1.0) * s_w),
                         (l - x_step / 2.0, points[-i - 1][-1] + h, i * s_w + s_w / 2.0),
                         (l - x_step / 2.0, points[-i - 2][-1] + h, (i + 1.0) * s_w + s_w / 2.0)))

        # Build front wall

        # Go through each column building bottom half triangles
        for i in range(points.shape[1]):
            tris.append(((i * x_step, h, 0.0),
                         ((i + 1.0) * x_step, h, 0.0),
                         (i * x_step + x_step / 2.0, points[-1][i] + h, s_w / 2.0)))
        # Go through each column building top half triangles
        for i in range(points.shape[1] - 1):
            tris.append((((i + 1) * x_step, h, 0.0),
                         ((i + 1) * x_step + x_step / 2.0, points[-1][i + 1] + h, s_w / 2.0),
                         (i * x_step + x_step / 2.0, points[-1][i] + h, s_w / 2.0)))

        # Build back wall

        # Go through each column building bottom half triangles
        for i in range(points.shape[1]):
            tris.append(((i * x_step, h, w),
                         ((i + 1.0) * x_step, h, w),
                         (i * x_step + x_step / 2.0, points[0][i] + h, w - s_w / 2.0)))
        # Go through each column building top half triangles
        for i in range(points.shape[1] - 1):
            tris.append((((i + 1) * x_step, h, w),
                         ((i + 1) * x_step + x_step / 2.0, points[0][i + 1] + h, w - s_w / 2.0),
                         (i * x_step + x_step / 2.0, points[0][i] + h, w - s_w / 2.0)))

        # Build top map. Split into squares along top surface, and build both triangles
        for i in range(points.shape[0] - 1):
            for j in range(points.shape[1] - 1):
                tris.append(((j * x_step + x_step / 2.0, points[-i - 1][j] + h, i * s_w + s_w / 2.0),
                             ((j + 1) * x_step + x_step / 2.0, points[-i - 2][j + 1] + h, (i + 1) * s_w + s_w / 2.0),
                             (j * x_step + x_step / 2.0, points[-i - 2][j] + h, (i + 1) * s_w + s_w / 2.0)))
                tris.append(((j * x_step + x_step / 2.0, points[-i - 1][j] + h, i * s_w + s_w / 2.0),
                             ((j + 1) * x_step + x_step / 2.0, points[-i - 1][j + 1] + h, i * s_w + s_w / 2.0),
                             ((j + 1) * x_step + x_step / 2.0, points[-i - 2][j + 1] + h, (i + 1) * s_w + s_w / 2.0)))

        return tris
