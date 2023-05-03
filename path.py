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


def calculate_joint_positions(path: LineString, n_links: int, link_length: float, offset: float) -> LineString:
    points = [path.interpolate(offset)]
    for n in range(n_links):
        # Find all points on path that are exactly one link length from the current joint
        intersects = path.intersection(points[-1].buffer(link_length).boundary)

        match intersects:
            case Point():
                points.append(intersects)

            case MultiPoint():
                # Approximate the position of the approximated point
                approx = path.interpolate(offset + (n + 1) * link_length)

                # Pick the intersection that is the closest to the approximated point
                next_point = min(intersects.geoms, key=lambda point: distance(approx, point))
                points.append(next_point)

    return LineString(points)


class SnakePath():
    path: LineString
    n_joints: int
    link_length: float

    def __init__(self, control_points, min_radius, n_joints, link_length):
        self.path = round_linestring(control_points, min_radius)
        self.n_joints = n_joints
        self.link_length = link_length

    def get_joint_angles(self, offset: float):
        return calculate_joint_positions(self.path, self.n_joints, self.link_length, offset)


control_points = LineString([(0, 0), (1, -0.2), (1, 1), (2, 0)])
path = round_linestring(control_points, 0.2)
joint_positions = calculate_joint_positions(path, 13, 0.2, 0)

plt.plot(*control_points.xy, '--D')
plt.plot(*path.xy)

print(joint_positions)
a = LineString(joint_positions)
border = a.buffer(WIDTH)

plt.plot(*a.xy, "o")
plt.plot(*border.boundary.xy)
plt.gca().set_aspect("equal")
plt.show()
