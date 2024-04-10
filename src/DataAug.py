###   l1 [xa, ya, xb, yb]   l2 [xa, ya, xb, yb]
def intersect(l1, l2):
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]

    temp = l1
    l1 = l2
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]

    if a*b < 0 and c*d < 0:
        return True
    else:
        return False


def does_line_intersect_rectangles(line_start, line_end):
    '''Check if a line segment intersects any of a list of rectangles'''

    rectangles = [
    [(1836.000000,8511.000000), (1333.000000,8722.000000)], 
    [(1355.000000,5964.000000), (1854.000000,5755.000000)], 
    [(5527.000000,5919.000000), (6017.000000,5712.000000)], 
    [(6072.000000,8523.000000), (5578.000000,8737.000000)], 
    [(7409.000000,8737.000000), (7899.000000,8535.000000)], 
    [(7868.000000,5902.000000), (7367.000000,5704.000000)], 
    [(13040.000000,5740.000000), (13546.000000,5937.000000)], 
    [(13518.000000,8544.000000), (13021.000000,8755.000000)], 
    ]

    for rect in rectangles:
        # Extract rectangle corners
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        # Generate rectangle edges
        rect_edges = [((x1, y1), (x1, y2)), ((x1, y2), (x2, y2)), 
                      ((x2, y2), (x2, y1)), ((x2, y1), (x1, y1))]
        for edge_start, edge_end in rect_edges:
            # if segments_intersect(line_start, line_end, edge_start, edge_end):
            if intersect(line_start + line_end, edge_start + edge_end):
                return True
    return False


# rectangles = [
#     [(1836.000000,8511.000000), (1333.000000,8722.000000)], 
#     [(1355.000000,5964.000000), (1854.000000,5755.000000)], 
#     [(5527.000000,5919.000000), (6017.000000,5712.000000)], 
#     [(6072.000000,8523.000000), (5578.000000,8737.000000)], 
#     [(7409.000000,8737.000000), (7899.000000,8535.000000)], 
#     [(7868.000000,5902.000000), (7367.000000,5704.000000)], 
#     [(13040.000000,5740.000000), (13546.000000,5937.000000)], 
#     [(13518.000000,8544.000000), (13021.000000,8755.000000)], 
# ]

# line_start = (5000, 1200)
# line_end = (10000, 10000)
# does_line_intersect_rectangles(line_start, line_end, rectangles)  # Returns True