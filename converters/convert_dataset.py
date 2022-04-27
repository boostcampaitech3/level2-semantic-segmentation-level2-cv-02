# Modified from
# https://stages.ai/competitions/191/discussion/talk/post/1326
# Special thanks for the original author Inseo Lee

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

import numpy as np

import os
import json
import shutil
import cv2


ROOT = '/opt/ml/input/data'

TRAIN_JSON = os.path.join(ROOT, 'train.json')
VAL_JSON = os.path.join(ROOT, 'val.json')
TEST_JSON = os.path.join(ROOT, 'test.json')

dir_lists = [
    os.path.join(ROOT, 'mmseg', 'images', 'train'),
    os.path.join(ROOT, 'mmseg', 'images', 'validation'),
    os.path.join(ROOT, 'mmseg', 'annotations', 'train'),
    os.path.join(ROOT, 'mmseg', 'annotations', 'validation'),
    os.path.join(ROOT, 'mmseg', 'test')
]

category_names = [
    'Backgroud',
    'General trash',
    'Paper',
    'Paper pack',
    'Metal',
    'Glass',
    'Plastic',
    'Styrofoam',
    'Plastic bag',
    'Battery',
    'Clothing'
]


def mkdir_or_exist(dir_path: str) -> None:
    """
    mmsegmentation format에 맞춰 폴더 생성
    """
    os.makedirs(dir_path, exist_ok=True)

    print(f'mkdir {dir_path} Success!')


def copy_images(json_path: str, target_path: str) -> None:
    """
    json_path에 있는 사진을 mmsegmentation format에 해당하는 폴더로 복사
    flag = train, val, test를 옵션으로 가진다.
    """
    with open(json_path, "r", encoding="utf8") as f:
        json_data = json.load(f)

    images = json_data["images"]
    for image in images:
        shutil.copyfile(os.path.join(ROOT, image["file_name"]), os.path.join(target_path, f"{image['id']:04}.jpg"))

    print(f'image copy to {target_path} Success!')


def create_mask(json_path: str, target_path: str) -> None:
    """
    dataset에 있는 annotations을 이용한 masks 생성
    """
    basic_transform = A.Compose([
        ToTensorV2()
    ])
    dataset = CustomDataLoader(
        data_dir=json_path,
        mode='train',
        transform=basic_transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    for _, masks, image_infos in dataloader:
        mask = masks[0].numpy()
        image_info = image_infos[0]
        cv2.imwrite(os.path.join(target_path, f"{image_info['id']:04}.png"), mask)

    print(f'create mask to {target_path} Success!')


def collate_fn(batch):
    return tuple(zip(*batch))


def get_classname(classID, cats) -> str:
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None) -> None:
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        images = cv2.imread(os.path.join(ROOT, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load categories into a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # Masks: size가 (height, width)인 2D
            # 각각의 pixel 값에는 category id를 할당
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos

        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        return len(self.coco.getImgIds())


if __name__ == '__main__':
    # Make directories
    for dir in dir_lists:
        mkdir_or_exist(dir)

    # Copy images
    copy_images(TRAIN_JSON, dir_lists[0])
    copy_images(VAL_JSON, dir_lists[1])
    copy_images(TEST_JSON, dir_lists[-1])

    # Create masks(annotations)
    create_mask(TRAIN_JSON, dir_lists[2])
    create_mask(VAL_JSON, dir_lists[3])
    print('Done!')
