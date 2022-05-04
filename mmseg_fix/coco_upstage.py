#
# boostcamp AI Tech
# Trash Semantic Segmentation Competition
#


# Insert this file into mmsegmentation/mmseg/datasets
# Modify __init__.py also

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class UpstageDataset(CustomDataset):
    """COCO Stuff style Upstage dataset."""

    CLASSES = (
        "Background", "General trash", "Paper", "Paper pack",
        "Metal", "Glass", "Plastic", "Styrofoam",
        "Plastic bag", "Battery", "Clothing"
    )

    # PALETTE = [
    #     [0, 0, 0],
    #     [192, 0, 128],
    #     [0, 128, 192],
    #     [0, 128, 64],
    #     [128, 0, 0],
    #     [64, 0, 128],
    #     [64, 0, 192],
    #     [192, 128, 64],
    #     [192, 192, 128],
    #     [64, 64, 128],
    #     [128, 0, 192]
    # ]
    PALETTE = [x for x in range(11)]

    def __init__(self, **kwargs):
        super().__init__(reduce_zero_label=False, **kwargs)
