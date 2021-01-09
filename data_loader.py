import os
import numpy as np
import pandas as pd
from settings import CELL_FEATURES_DIR, SURVIVAL_DATA_DIR


class DatasetLoader:

    def __init__(self) -> None:
        self.patient_id_list = []
        self.features = None
        self.survival_data = None

    def load_cell_features(self, root_dir, save_path=None):
        """使用描述分布的4大特征来构造 每个患者的特征
        """
        root_dir = os.path.join(*[root_dir, CELL_FEATURES_DIR, 'smoothfeatures50'])

        features_list = []
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.xlsx'):
                full_xlsx_file_name = os.path.join(root_dir, file_name)
                print(full_xlsx_file_name)
                self.patient_id_list.append(file_name.split('.')[0])
                df = pd.read_excel(full_xlsx_file_name, engine='openpyxl')

                # 计算每一列的均值方差
                df_without_co = df[df.columns.tolist()[2:]]
                features_list.append(np.concatenate([df_without_co.mean().values, df_without_co.std().values]))
        
        self.features = np.array(features_list)
        if save_path:
            np.savetxt(save_path, self.features, delimiter=",")
        return self.features

    def load_cell_features_from_path(self, save_path):
        self.features = np.loadtxt(save_path, delimiter=",")
        return self.features

    def load_survial_data(self, root_dir, save_path=None):
        survival_data_dir = os.path.join(*[root_dir, SURVIVAL_DATA_DIR, 'survival_data.xlsx'])
        df = pd.read_excel(survival_data_dir, engine='openpyxl')
        self.survival_data = df[df['ID'].isin(self.patient_id_list)][df.columns.tolist()[1:]].values
        if save_path:
            np.savetxt(save_path, self.survival_data, delimiter=",")
        return self.survival_data
        