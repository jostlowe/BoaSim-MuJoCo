import math
from dataclasses import dataclass
from shapely import LineString, Point, MultiPoint, distance
import matplotlib.pyplot as plt
import numpy as np

MIN_RADIUS = 0.2
LINK_LENGTH = 0.2
WIDTH = 0.08
L_0 = 0.1


def round_linestring(linestring: LineString, radius: float) -> LineString:
    return linestring.offset_curve(radius).offset_curve(-2 * radius).offset_curve(radius)


def three_point_angle(a, b, c):
    ab, bc = b-a, c-b
    theta_1 = np.arctan2(ab[0], ab[1])
    theta_2 = np.arctan2(bc[0], bc[1])
    return theta_1-theta_2


class SnakePath:
    path: LineString
    control_points: LineString
    n_links: int
    link_length: float

    def __init__(self, control_points, min_radius, n_links, link_length):
        self.control_points = control_points
        self.path = round_linestring(control_points, min_radius)
        self.n_links = n_links
        self.link_length = link_length

    def get_vertices(self, offset):
        points = [self.path.interpolate(offset)]
        for n in range(self.n_links):
            # Find all points on path that are exactly one link length from the current joint
            intersects = self.path.intersection(points[-1].buffer(self.link_length).boundary)

            match intersects:
                case Point():
                    points.append(intersects)

                case MultiPoint():
                    # Approximate the position of the approximated point
                    approx = self.path.interpolate(offset + (n + 1) * self.link_length)

                    # Pick the intersection that is the closest to the approximated point
                    next_point = min(intersects.geoms, key=lambda point: distance(approx, point))
                    points.append(next_point)

        return LineString(points)

    def get_joint_angles(self, offset):
        vertices = np.array(self.get_vertices(offset).xy).T
        return [three_point_angle(a, b, c) for a, b, c in zip(vertices, vertices[1:], vertices[2:])]


path = SnakePath(
    control_points=LineString([(0, 0), (2.2, 0), (2.2, 0.34), (1.45, 0.34), (1.45, 0.66), (3.0, 0.66)]),
    min_radius=0.15,
    n_links=13,
    link_length=0.2
)


plt.plot(*path.control_points.xy, '--D')
plt.plot(*path.path.xy)

print(path.get_vertices(1.0))
print(path.get_joint_angles(1))
a = LineString(path.get_vertices(1))
border = a.buffer(WIDTH)

plt.plot(*a.xy, "o")
plt.plot(*border.boundary.xy)
plt.plot(*path.path.buffer(WIDTH).boundary.xy)

obstacles = [
    (1.2, 0.17),
    (1, -0.17),
    (2, 0.17),
    (1.7, -0.17),
    (1.65, 0.5),
    (2, 0.82),
    (2.5, 0.5)
]

for obstacle in obstacles:
    geom = Point(obstacle).buffer(0.08).boundary.xy
    plt.plot(*geom, 'k')

plt.gca().set_aspect("equal")
plt.show()

