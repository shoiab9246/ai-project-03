#!/usr/local/bin/python3
#
# Authors: [PLEASE PUT YOUR NAMES AND USER IDS HERE]
#
# Alex Shroyer , Shoiab Mohammed
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, April 2021
from PIL import Image
from scipy.ndimage import filters
import imageio
import numpy as np
import sys

# calculate "Edge strength map" of an image
def edge_strength(input_image):
    grayscale = np.array(input_image.convert('L'))
    filtered_y = np.zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return np.sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image


# main program
gt_row = -1
gt_col = -1
if len(sys.argv) == 2:
    input_filename = sys.argv[1]
elif len(sys.argv) == 4:
    (input_filename, gt_row, gt_col) = sys.argv[1:]
else:
    raise Exception("Program requires either 1 or 3 parameters")

output_filename = input_filename.split('/')[-1]

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edges = edge_strength(input_image)
# heuristic increases edge intensity around upper 1/3 of image
edges *= np.cos(np.linspace(-0.8, 1.5, edges.shape[0]))[:,None].repeat(edges.shape[1], axis=1)**5
imageio.imwrite('edges.jpg', np.uint8(255 * edges / np.amax(edges)))

# part 1 (naive bayes)
ridge = np.argmax(edges, axis=0)
img = draw_edge(input_image, ridge, (255, 0, 0), 5)

# part 2 (HMM)
# most mountainy row heuristic: people tend to put the mountain in the center of the image.
mean_mtn_row = np.mean(ridge[int(len(ridge)*0.4):int(len(ridge)*0.6)], axis=0)
# img = draw_edge(img, [mean_mtn_row] * edges.shape[1], (0,0,0), 5) # draw the mean row in black
# mdiff = edges * -np.log((edges - mean_mtn_row)**2 +1e-200)
# imageio.imwrite("diff.jpg", np.uint8(255 * mdiff / np.amax(mdiff)))

half_width = edges.shape[1] // 2
current_row = round(mean_mtn_row)
hmm_row = np.zeros(edges.shape[1])
sigma2 = 1**2
# print(mean_mtn_row)

# center to right
for i in range(half_width, edges.shape[1]):
    # hmm_row[i] = current_row
    current_col = edges[:,i].copy()
    possible_rows = np.argsort(current_col)[-5:] # row locations of 5 brightest pixels in column
    hmm_row[i] = min([(edges[row,i]*(row-current_row)**2/sigma2,row) for row in possible_rows])[1]

# center to left
for i in range(half_width-1,-1,-1):
    # hmm_row[i] = current_row
    current_col = edges[:,i].copy()
    possible_rows = np.argsort(current_col)[-5:] # row locations of 5 brightest pixels in column
    hmm_row[i] = min([(edges[row,i]*(row-current_row)**2/sigma2,row) for row in possible_rows])[1]
    # compute probability for each pixel in edges[:,i]
    # lower probs for pixels which are farther from current_row
    # hmm_row[i] = max(np.exp((possible_rows - current_row)**2 / sigma2))
# print(hmm_row)

draw_edge(img, hmm_row, (0,255,0), 3)


imageio.imwrite(f"output/{output_filename}", img)
