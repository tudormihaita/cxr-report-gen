import os
import numpy as np
import pandas as pd

from constants import DATASETS_PATH, CHEXPERT_LABELS
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class IUXrayPreprocessor:
    def __init__(self, data_path=DATASETS_PATH['iu-xray']):
        self.iu_xray = None
        self.data_path = data_path
        self.reports = pd.read_csv(os.path.join(self.data_path, 'indiana_reports.csv'))
        self.annotations = pd.read_csv(os.path.join(self.data_path, 'indiana_annotations.csv'))
        self.projections = pd.read_csv(os.path.join(self.data_path, 'indiana_projections.csv'))

    def preprocess(self):
        self._clean_data()
        self._merge_data()
        self._assign_splits()
        self._generate_prompts()
        self.iu_xray = self._finalize()
        return self.iu_xray

    def _clean_data(self):
        self.reports.drop(columns=["image", "indication", "comparison"], inplace=True, errors='ignore')
        self.annotations.fillna(0, inplace=True)

    def _merge_data(self):
        frontal_proj = self.projections[self.projections['projection'] == 'Frontal'][['uid', 'filename']]
        lateral_proj = self.projections[self.projections['projection'] == 'Lateral'][['uid', 'filename']]

        frontal_proj = frontal_proj.drop_duplicates(subset='uid', keep='first').rename(
            columns={'filename': 'frontal_filename'})
        lateral_proj = lateral_proj.drop_duplicates(subset='uid', keep='first').rename(
            columns={'filename': 'lateral_filename'})

        self.iu_xray = self.reports.merge(frontal_proj, on="uid", how="left")
        self.iu_xray = self.iu_xray.merge(lateral_proj, on="uid", how="left")

        self.iu_xray = self.iu_xray.merge(self.annotations, on="uid", how="left")

    def _assign_splits(self):
        """
        Assign train, validation and test splits to the dataset using stratified sampling based on the labels.
        The dataset is split into 5 folds, and then the first fold is used for training, the second for validation,
        and the third for testing.
        Percentages:
        """
        X = self.iu_xray[['uid']]
        Y = self.iu_xray[CHEXPERT_LABELS].replace(-1, 0).values

        mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        folds =  list(mskf.split(X, Y))
        train_idx = np.concatenate([folds[i][1] for i in range(8)])
        temp_idx = np.concatenate([folds[i][1] for i in range(8, 10)])

        X_temp = self.iu_xray.iloc[temp_idx]
        y_temp = Y[temp_idx]

        val_idx, test_idx = next(MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(X_temp, y_temp))

        train_split, val_split, test_split = X.iloc[train_idx], X_temp.iloc[val_idx], X_temp.iloc[test_idx]
        self.iu_xray['split'] = 'none'
        self.iu_xray.loc[train_split.index, 'split'] = 'train'
        self.iu_xray.loc[val_split.index, 'split'] = 'val'
        self.iu_xray.loc[test_split.index, 'split'] = 'test'

    def _generate_prompts(self):
        def generate_prompt(row):
            labels = [label for label in CHEXPERT_LABELS if row[label] == 1.0]
            if labels:
                return "This chest X-ray shows " + ", ".join(labels) + "."
            else:
                return "This chest X-ray shows no abnormal findings."

        self.iu_xray['prompt'] = self.iu_xray.apply(generate_prompt, axis=1)

    def _finalize(self):
        cols = ['uid', 'split', 'frontal_filename', 'lateral_filename', 'report',
                'prompt', 'findings', 'impression', 'MeSH', 'Problems'] + CHEXPERT_LABELS
        return self.iu_xray[cols]