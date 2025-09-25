# load a data_generation-0.txt file 
# and visualise the first 10 data points

import matplotlib.pyplot as plt
import argparse
from utils import decode

if __name__ == "__main__":
    # argparse for filename
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, default="data_generation-0.txt")
    args = argparser.parse_args()
    
    filename = args.file

    with open(filename, "r") as f:
        lines = f.readlines()
        # strip newlines
        lines = [line.strip() for line in lines if line.strip()]
        lines = [ [int(x) for x in line.split(",") if x.strip()] for line in lines]

    # take 10 lines 
    lines = lines[:10]
    print(f"Decoded first 10 lines {lines}")


    decoded = map(decode, lines)


    # decode each line into a list of integers
    list_of_points = [ p for _v,p,_k in decoded ]

    # each entry is a list of n points
    # plot everything all on the same plot 
    plt.figure()
        
    plt.figure(figsize=(10, 8))
    
    for i, points in enumerate(list_of_points):
        # Extract x and y coordinates
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        
        # Plot the simplex (close the polygon by adding first point at end)
        plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 
                marker='o', linestyle='-', alpha=0.7, label=f'Simplex {i+1}')
            
    plt.title('First 10 Data Series from ' + filename)
    plt.xlabel('Point Index')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()