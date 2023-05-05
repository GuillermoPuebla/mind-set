class SizeMonitor:
    def __init__(self, window=3):
        self.window = window
        self.sizes = []

    def add(self, size):
        self.sizes.append(size)

    def check_stable(self):
        # if the last n are the same and different from the initial number
        return len(set(self.sizes[-self.window :])) == 1 and self.sizes[-1] != self.sizes[0]


def chaikins_corner_cutting(points, num_iterations):
    for _ in range(num_iterations):
        new_points = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            q = ((3 / 4) * p1[0] + (1 / 4) * p2[0], (3 / 4) * p1[1] + (1 / 4) * p2[1])
            r = ((1 / 4) * p1[0] + (3 / 4) * p2[0], (1 / 4) * p1[1] + (3 / 4) * p2[1])

            new_points.extend([q, r])
        points = new_points
    return points
