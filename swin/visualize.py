#
# boostcamp AI Tech
# Trash Semantic Segmentation Competition
#


import numpy as np
import pandas as pd

import argparse
import cv2


PALETTE = [
    [0, 0, 0],
    [192, 0, 128],
    [0, 128, 192],
    [0, 128, 64],
    [128, 0, 0],
    [64, 0, 128],
    [64, 0, 192],
    [192, 128, 64],
    [192, 192, 128],
    [64, 64, 128],
    [128, 0, 192]
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    parser.add_argument('image_id', type=int)
    args = parser.parse_args()

    predictions = pd.read_csv(f"./{args.file_name}.csv")
    predictions = predictions['PredictionString'][args.image_id - 1].split(' ')

    assert len(predictions) == 256 * 256

    mask = np.zeros((256, 256, 3))
    for i in range(256):
        for j in range(256):
            for rgb in range(3):
                mask[i][j][rgb] = PALETTE[int(predictions[i * 256 + j])][rgb]

    cv2.imwrite(f"./visualization/{args.file_name}_image{args.image_id}.png", mask)
