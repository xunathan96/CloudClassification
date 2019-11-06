import numpy as np
from skimage import io
import csv

# SHAPE OF TRAINING IMAGES
num_rows = 1400
num_cols = 2100

# CONVERTS THE ENCODED PIXEL AND RUNLENGTH INTO ROW/COL COORDINATES
def pixel_to_coord(start_pixel, run_len):
    row = (start_pixel-1) % num_rows
    col = (start_pixel-1) // num_rows

    row_array = np.arange(row, row + run_len, dtype=int)
    col_array = col * np.ones(run_len, dtype=int)
    return row_array, col_array

# DECODES THE ENCODED PIXELS OF THE SEGMENTATION MASK
# Returns a row coordinate array and col coordinate array
# corresponding to the (row, column) pixels of the mask
def decode_pixels(encoded_pixels):
    row_coords = np.array([], dtype=int)
    col_coords = np.array([], dtype=int)

    data = encoded_pixels.split()
    for i in range(0, len(data)-1, 2):
        start_pixel = int(data[i])
        run_len = int(data[i+1])

        row_array, col_array = pixel_to_coord(start_pixel, run_len)
        row_coords = np.append(row_coords, row_array)
        col_coords = np.append(col_coords, col_array)

    return row_coords, col_coords

# LOADS THE MASK OF THE TRAINING IMAGES INTO A DICTIONARY
# image_mask[image_label][rows] := rows corresponding to the image masks
# image_mask[image_label][cols] := cols corresponding to the image masks
def load_seg_masks():
    image_mask = {}
    with open('train_small.csv', 'r') as csvFile:
        next(csvFile) #skip first line
        reader = csv.reader(csvFile)
        for row in reader:
            image_label = row[0]
            encoded_pixels = row[1]
            row_coords, col_coords = decode_pixels(encoded_pixels)
            image_mask[image_label] = {'rows': row_coords, 'cols': col_coords}

    csvFile.close()
    return image_mask



def main():
    # LOAD IMAGE MASKS
    image_mask = load_seg_masks()
    # STORE ROW/COL OF MASKS into arrays
    row_coords = image_mask['0011165.jpg_Flower']['rows']
    col_coords = image_mask['0011165.jpg_Flower']['cols']

    image = io.imread('0011165.jpg')

    # COLOR MASK GREEN
    image[row_coords, col_coords] = (0,255,0)

    io.imshow(image)
    io.show()



if __name__ == '__main__':
    main()
