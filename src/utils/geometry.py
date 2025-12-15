def ccw(A, B, C):
    """
    Check if three points are listed in counter-clockwise order.
    A, B, C are tuples (x, y).
    """
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segments_intersect(p1, p2, p3, p4):
    """
    Check if line segment (p1, p2) intersects with line segment (p3, p4).
    p1, p2, p3, p4 are tuples (x, y).
    """
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def is_point_in_polygon(point, polygon):
    """
    Check if point (x, y) is inside the polygon (list of points).
    Ray casting algorithm.
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
