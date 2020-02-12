import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def str_to_img(row):
    return np.array(list(map(lambda x: int(x), row.split(' ')))).reshape(
        (48, 48))


class FacialExpressionDataset(ImageFolder):
    """
    ImageFolder dataset with an additional class contructor that allows to generate the correct folder - image structure for `ImageFolder`
    """
    label_to_name = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    @classmethod
    def from_csv(cls,
                 file_path: str,
                 out_dir: str,
                 *args,
                 overwrite=False,
                 **kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        if not out_dir.exists() or overwrite:
            df = pd.read_csv(file_path)
            bar = tqdm(total=len(df))

            df['label'] = df.emotion.apply(lambda x: cls.label_to_name[x])
            df['img'] = df.pixels.apply(str_to_img)

            def store_img_from_row(row: pd.Series):
                """Use the label to store the image in the correct subfolder
                
                :param row: A row of the dataframe
                :type row: pd.Series
                """
                file_dir = out_dir / row['label']
                file_dir.mkdir(exist_ok=True)
                filename = file_dir / f'{row.name}.png'
                img_g = row['img']

                cv2.imwrite(str(filename), img_g)
                bar.update()

            df.apply(store_img_from_row, axis=1)

        return cls(root=str(out_dir), *args, **kwargs)
