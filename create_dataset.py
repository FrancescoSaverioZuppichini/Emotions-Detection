from Project import Project
from data import FacialExpressionDataset

FacialExpressionDataset.from_csv(Project().data_dir / 'train.csv', Project().data_dir / 'train', overwrite=True)