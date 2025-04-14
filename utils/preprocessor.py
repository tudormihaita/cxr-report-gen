import os
import re
import logging
import numpy as np
import pandas as pd

from utils import section_parser as sp
from constants import DATASETS_PATH, MIMIC_IMAGES_PATH, MIMIC_REPORTS_PATH, CHEXPERT_LABELS
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IUXrayPreprocessor:
    def __init__(self, data_path=DATASETS_PATH['iu-xray']):
        self.iu_xray = None
        self.data_path = data_path
        self.reports = pd.read_csv(os.path.join(self.data_path, 'indiana_reports.csv'))
        self.annotations = pd.read_csv(os.path.join(self.data_path, 'indiana_annotations.csv'))
        self.projections = pd.read_csv(os.path.join(self.data_path, 'indiana_projections.csv'))

    def preprocess(self) -> pd.DataFrame:
        self._clean_data()
        self._merge_data()
        self._assign_splits()
        self._generate_prompts()
        self.iu_xray = self._finalize()
        return self.iu_xray

    def _clean_data(self) -> None:
        self.reports.drop(columns=["image", "indication", "comparison"], inplace=True, errors='ignore')
        self.annotations.fillna(0, inplace=True)

    def _merge_data(self) -> None:
        frontal_proj = self.projections[self.projections['projection'] == 'Frontal'][['uid', 'filename']]
        lateral_proj = self.projections[self.projections['projection'] == 'Lateral'][['uid', 'filename']]

        frontal_proj = frontal_proj.drop_duplicates(subset='uid', keep='first').rename(
            columns={'filename': 'frontal_filename'})
        lateral_proj = lateral_proj.drop_duplicates(subset='uid', keep='first').rename(
            columns={'filename': 'lateral_filename'})

        self.iu_xray = self.reports.merge(frontal_proj, on="uid", how="left")
        self.iu_xray = self.iu_xray.merge(lateral_proj, on="uid", how="left")

        self.iu_xray = self.iu_xray.merge(self.annotations, on="uid", how="left")

    def _assign_splits(self) -> None:
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

    def _generate_prompts(self) -> None:
        def generate_prompt(row):
            labels = [label for label in CHEXPERT_LABELS if row[label] == 1.0]
            if labels:
                return "This chest X-ray shows " + ", ".join(labels) + "."
            else:
                return "This chest X-ray shows no abnormal findings."

        self.iu_xray['prompt'] = self.iu_xray.apply(generate_prompt, axis=1)

    def _finalize(self) -> pd.DataFrame:
        cols = ['uid', 'split', 'frontal_filename', 'lateral_filename', 'report',
                'prompt', 'findings', 'impression', 'MeSH', 'Problems'] + CHEXPERT_LABELS
        return self.iu_xray[cols]


class MimicCXRPreprocessor:
    def __init__(self, data_path=DATASETS_PATH['mimic-cxr']):
        self.mimic_cxr = None
        self.data_path = data_path
        self.chexpert_labels = pd.read_csv(os.path.join(self.data_path, 'mimic-cxr-2.0.0-chexpert.csv.gz'), compression='gzip')
        self.mimic_splits = pd.read_csv(os.path.join(self.data_path, 'mimic-cxr-2.0.0-split.csv.gz'), compression='gzip')

        self.images_path = os.path.join(str(self.data_path), MIMIC_IMAGES_PATH)
        self.reports_path = os.path.join(str(self.data_path), MIMIC_REPORTS_PATH)

    def preprocess(self):
        self._clean_data()
        self.mimic_cxr = self._find_image_report_pairs()
        self.mimic_cxr = self._extract_sections(self.mimic_cxr)
        self.mimic_cxr = self._assign_splits(self._find_labels(self.mimic_cxr))
        return self.mimic_cxr

    def _clean_data(self) -> None:
        self.chexpert_labels.fillna(0, inplace=True)

    def _find_image_report_pairs(self) -> pd.DataFrame:
        data = []

        for p_folder in sorted(os.listdir(self.images_path)):
            p_path = os.path.join(str(self.images_path), p_folder)
            if not os.path.isdir(p_path):
                continue

            for subject_id in os.listdir(p_path):
                subject_path = os.path.join(p_path, subject_id)
                if not os.path.isdir(subject_path):
                    continue

                for study_id in os.listdir(subject_path):
                    study_path = os.path.join(subject_path, study_id)
                    if not os.path.isdir(study_path):
                        continue

                    report_txt_file = os.path.join(self.reports_path, p_folder, subject_id, f'{study_id}.txt')
                    if not os.path.isfile(report_txt_file):
                        logger.warning(f'Missing report file: {report_txt_file}')
                        continue

                    for image_filename in os.listdir(study_path):
                        if not image_filename.endswith('.jpg'):
                            continue

                        dicom_id = os.path.splitext(image_filename)[0]
                        image_path = os.path.join(p_folder, subject_id, study_id, image_filename)
                        report_path = os.path.join(p_folder, subject_id, f'{study_id}.txt')

                        data.append({
                            'dicom_id': dicom_id,
                            'study_id': study_id,
                            'subject_id': subject_id,
                            'image_path': image_path,
                            'report_path': report_path
                        })
        return pd.DataFrame(data)

    def _extract_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        custom_names, custom_indices = sp.custom_mimic_cxr_rules()

        findings_list, impressions_list = [], []
        for _, row in df.iterrows():
            findings, impression = '', ''
            try:
                report_path = os.path.join(self.reports_path, row['report_path'])
                with open(report_path, 'r') as f:
                    text = f.read()

                    study_id = row['study_id']
                    if study_id in custom_indices:
                        idx = custom_indices[study_id]
                        relevant_text = text[idx[0]:idx[1]].strip()
                        findings = relevant_text
                    else:
                        sections, section_names, _ = sp.section_text(text)

                        last_text = sections[-1].strip() if sections else ''
                        section_map = {name: content.strip() for name, content in zip(section_names, sections)}
                        if 'findings' in section_map:
                            findings = section_map['findings']
                        if 'impression' in section_map:
                            impression = section_map['impression']

                        if not findings and not impression and last_text:
                            findings = last_text
            except Exception as e:
                logger.error(f"Error parsing report {row['report_path']}: {e}")

            findings_list.append(findings)
            impressions_list.append(impression)

        df['findings'] = findings_list
        df['impression'] = impressions_list
        return df

    def _find_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        self.chexpert_labels['subject_id'] = self.chexpert_labels['subject_id'].apply(lambda x: f'p{x}')
        self.chexpert_labels['study_id'] = self.chexpert_labels['study_id'].apply(lambda x: f's{x}')

        merged_df = pd.merge(
            df,
            self.chexpert_labels,
            how='inner',
            on=['subject_id', 'study_id']
        )

        merged_keys = set(zip(merged_df['subject_id'], merged_df['study_id']))
        df_keys = set(zip(df['subject_id'], df['study_id']))

        missing_keys = df_keys - merged_keys
        print(f"Missing study_ids (not in chexpert labels): {missing_keys}")

        return merged_df

    def _assign_splits(self, df: pd.DataFrame, val_ratio: float = 0.15) -> pd.DataFrame:
        self.mimic_splits['subject_id'] = self.mimic_splits['subject_id'].apply(lambda x: f'p{x}')
        self.mimic_splits['study_id'] = self.mimic_splits['study_id'].apply(lambda x: f's{x}')

        merged_df = pd.merge(
            df,
            self.mimic_splits,
            how='inner',
            on=['dicom_id', 'subject_id', 'study_id']
        )

        test_split = merged_df[merged_df['split'] == 'test'].copy()
        trainval_df = merged_df[merged_df['split'] == 'train'].copy()

        Y = trainval_df[CHEXPERT_LABELS].replace(-1, 0).values
        X = trainval_df[['dicom_id']]

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
        train_idx, val_idx = next(splitter.split(X, Y))

        train_split = trainval_df.iloc[train_idx].copy()
        val_split = trainval_df.iloc[val_idx].copy()

        train_split['split'] = 'train'
        val_split['split'] = 'val'

        merged_df = pd.concat([train_split, val_split, test_split], ignore_index=True)
        return merged_df