# Modified from
# https://stages.ai/competitions/191/discussion/talk/post/1331
# Special thanks for the original author Jeongjae Park

import pandas as pd
from tqdm import tqdm
import os

# Prediction file names should be {mIoU score}.csv
output_list = os.listdir('./output_csv')
output_list.sort(reverse=True)
print(output_list)

df_list = []
for output in output_list:
    df_list.append(pd.read_csv(f'./output_csv/{output}'))

submission = pd.DataFrame()
submission['image_id'] = df_list[0]['image_id']

PredictionString = []
for image_idx in tqdm(range(len(df_list[0]))):
    pixel_list = []
    for model_idx in range(len(df_list)):
        prediction_string = df_list[model_idx]['PredictionString'][image_idx].replace('  ', ' ').split(' ')
        while prediction_string[0] == '':
            prediction_string = prediction_string[1:]
        pixel_list.append(prediction_string)

    result = ''

    for pixel_idx in range(len(pixel_list[0])):
        pixel_count = {
            '0': 0, '1': 0, '2': 0, '3': 0,
            '4': 0, '5': 0, '6': 0, '7': 0,
            '8': 0, '9': 0, '10': 0
        }

        for model_idx in range(len(pixel_list)):
            pixel_count[pixel_list[model_idx][pixel_idx]] += 1

        voted_pixel = [key for key, value in pixel_count.items() if value == max(pixel_count.values())]

        if len(voted_pixel) == 1:
            result += voted_pixel[0] + ' '
        else:
            for model_idx in range(len(pixel_list)):
                pixel_candidate = pixel_list[model_idx][pixel_idx]

                if pixel_candidate in voted_pixel:
                    result += pixel_candidate + ' '
                    break

    result = result[:-1]
    PredictionString.append(result)

submission['PredictionString'] = PredictionString
submission.to_csv('./ensemble.csv', index=False)
