import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import decode


def convex_hull_2d(points):
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(set(tuple(p) for p in points))
    if len(points) <= 1:
        return points

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def sort_points_angular(points):
    if len(points) <= 2:
        return points

    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Calculate angles from centroid
    angles = []
    for point in points:
        angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        angles.append(angle)

    # Sort points by angle
    sorted_indices = np.argsort(angles)
    return [points[i] for i in sorted_indices]


def sort_points_for_polygon(points):
    if len(points) <= 2:
        return points

    try:
        hull_points = convex_hull_2d(points)
        if len(hull_points) == len(points):
            # all points are on the convex hull
            return hull_points
        else:
            # some points are interior, fall back to angular sort
            return sort_points_angular(points)
    except:
        #  angular sort
        return sort_points_angular(points)


def calculate_polygon_area(points):
    if len(points) < 3:
        return 0

    n = len(points)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2


if __name__ == "__main__":
    # argparse for filename
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, default="data_generation-0.txt")
    argparser.add_argument(
        "--method",
        type=str,
        choices=["hull", "angular", "auto"],
        default="auto",
        help="Point ordering method",
    )
    argparser.add_argument(
        "--show_points", action="store_true", help="Show individual points as dots"
    )
    argparser.add_argument(
        "--show_centroid", action="store_true", help="Show polygon centroids"
    )
    args = argparser.parse_args()

    filename = args.file

    with open(filename, "r") as f:
        lines = f.readlines()
        # strip newlines
        lines = [line.strip() for line in lines if line.strip()]
        lines = [[int(x) for x in line.split(",") if x.strip()] for line in lines]

    # take 10 lines
    lines = lines[:10]
    print(f"Loaded first 10 lines")

    decoded = map(decode, lines)

    # decode each line into a list of points
    list_of_points = [(v, p) for v, p, _k in decoded]

    # Create visualization
    plt.figure(figsize=(12, 10))

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(list_of_points)))

    for i, (v, points) in enumerate(list_of_points):
        # weight points by v
        # points = [(v[j] * pt[0].item(), v[j] * pt[1].item()) for j, pt in enumerate(points) if v[j] != 0]

        if len(points) < 2:
            continue

        # Convert to numpy array for easier handling
        points = np.array(points)

        # Sort points to form a proper polygon
        if args.method == "hull":
            sorted_points = convex_hull_2d(points)
        elif args.method == "angular":
            sorted_points = sort_points_angular(points)
        else:  # auto
            sorted_points = sort_points_for_polygon(points)

        sorted_points = np.array(sorted_points)

        # Extract coordinates
        x_coords = sorted_points[:, 0]
        y_coords = sorted_points[:, 1]

        # Calculate some statistics
        area = calculate_polygon_area(sorted_points)
        centroid = np.mean(sorted_points, axis=0)

        # Plot the polygon (close it by adding first point at end)
        plt.plot(
            np.append(x_coords, x_coords[0]),
            np.append(y_coords, y_coords[0]),
            color=colors[i],
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label=f"Polytope {i + 1} (Area: {area:.1f})",
        )

        # Fill the polygon
        plt.fill(
            np.append(x_coords, x_coords[0]),
            np.append(y_coords, y_coords[0]),
            color=colors[i],
            alpha=0.2,
        )

        # Show individual points if requested
        if args.show_points:
            plt.scatter(x_coords, y_coords, color=colors[i], s=50, zorder=5)
            # Number the points to see the ordering
            for j, (x, y) in enumerate(sorted_points):
                plt.annotate(
                    str(j),
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=colors[i],
                )

        # Show centroid if requested
        if args.show_centroid:
            plt.scatter(
                centroid[0], centroid[1], color=colors[i], marker="x", s=100, zorder=6
            )

        print(f"Polytope {i + 1}: {len(sorted_points)} vertices, area = {area:.2f}")

    plt.title(f"Polytopes from {filename} (Method: {args.method})")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")  # Equal aspect ratio for proper shape visualization
    plt.tight_layout()

    print(f"\nVisualization complete. Used {args.method} method for point ordering.")
    print("Try different methods with --method hull/angular/auto")
    print("Use --show_points to see vertex numbering")
    print("Use --show_centroid to see polygon centers")

    plt.show()
