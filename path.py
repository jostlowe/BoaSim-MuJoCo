from dataclasses import dataclass
import shapely
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class LineArcPath:
    control_points: [(float, float)]

path = LineArcPath([
    (0,0),
    (1,0),
    (2,1),
    (3,1)
])

def round(linestring, radius):
    return linestring.offset_curve(radius).offset_curve(-2*radius).offset_curve(radius)

seg = shapely.LineString([(0,0), (1,0), (1,1), (2,0)])
plt.plot(*seg.xy)
rounded = round(seg, 0.2)
plt.plot(*rounded.xy)

start_point = rounded.interpolate(0)
points = [start_point]

for n in range(13):
    rad = points[-1].buffer(0.2).boundary
    intersects = rounded.intersection(rad)

    if isinstance(intersects, shapely.Point):
        points.append(intersects)

    if isinstance(intersects, shapely.MultiPoint):
        current = points[-1]
        prev = points[-2]
        next_point = next(filter(lambda i: not i.equals_exact(prev, tolerance=0.01), intersects.geoms))
        points.append(next_point)


print(points)

a = np.array([(p.x, p.y) for p in points])
print(a)
plt.plot(a[:, 0], a[:, 1], "-o")
plt.gca().set_aspect('equal')
plt.show()

a = shapely.LineString(path.control_points)

