import pandas as pd
from timeStampToDate import to_date
import os
import glob
from pathlib import Path
import random as rnd
import numpy as np
from sklearn.utils import shuffle


class CreateDataset:

    # Initializer / Instance Attributes
    def __init__(self, f_path, space_limit_dic=None, feature_list=None, label=None, attack_type=None):
        if space_limit_dic is None:
            self.space_limit_dic = {
                "lat_min": 36,
                "lat_max": 70,
                "lon_min": 10,
                "lon_max": 80
            }

        if feature_list is None:
            feature_list = ["icao24", "time", "lat", "lon", "heading", "velocity", "baroaltitude", "geoaltitude",
                            "label"]

        # constant
        # velocity
        # random
        # constant velocity
        # all
        if attack_type is None:
            attack_type = "constant"

        self.f_path = f_path
        self.space_limit_dic = space_limit_dic
        self.feature_list = feature_list
        self.label = label
        self.attack_type = attack_type

    def get_dataset(self) -> pd.DataFrame:
        
        if self.label == 0:
            # res_path = str(self.f_path) + "resultData\\" + str(self.attack_type) + "_attack_dataframe.csv"
            res_path = "C:\\Users\\nursa\\Documents\\Nursah\\TİK 3\\Flight\\VsCode\\data\\resultData\\"  + str(self.attack_type) + "_attack_dataframe.csv"
        else:
            res_path = 'C:\\Users\\nursa\\Documents\\Nursah\\TİK 3\\Flight\\VsCode\\data\\resultData\\flight_dataframe.csv'
            # res_path = str(self.f_path) + "resultData\\" + "flight_dataframe.csv"

        if self.check_dataframe(res_path) == 0:
            # df = self.merge_datasets()
            # df = self.select_airspace(df)
            #df = self.select_features(df)
            if self.label == 0:
                print("deneme")
                df = pd.read_csv("C:\\Users\\nursa\\Documents\\Nursah\\TİK 3\\Flight\\VsCode\\data\\resultData\\flight_to_attack_dataframe.csv")
                df = self.create_attack(df)
                df["label"] = self.label
        else:
            df = pd.read_csv(res_path)

        print("Dosyaya yazma islemi gerceklestiriliyor...")
        df.to_csv(res_path, index=False)

        return df

    @staticmethod
    def check_dataframe(res_path) -> bool:
        check = Path(res_path)

        if check.is_file():
            return True
        return False

    def merge_datasets(self) -> pd.DataFrame:
        print("Ham veri kumeleri birlestiriliyor...")
        df = pd.read_csv(str(self.f_path))
        # res_path = str(self.f_path) + "rawData\\"
        # print(res_path)
        # csv_files = glob.glob(os.path.join(res_path, "*.csv"))

        # file_dict = {}
        # counter = 0
        # for file in csv_files:
        #    df = pd.read_csv(file)
        #    file_dict[counter] = df
        #    counter += 1

        # df = pd.concat(file_dict)
        df.reset_index(drop=True, inplace=True)

        df.sort_values("time")
        df["label"] = self.label

        return df

    def select_airspace(self, df) -> pd.DataFrame:
        print("Hava sahasi seciliyor...")
        df_selected_space = df[(df.lon < 100) & (df.lon > 10)
                             & (df.lat < 100) & (df.lat > 10)]

#       df_selected_space = df[(df.lon < self.space_limit_dic["lon_max"]) & (df.lon > self.space_limit_dic["lon_min"])
#                             & (df.lat < self.space_limit_dic["lat_max"]) & (df.lat > self.space_limit_dic["lat_min"])]
        df_selected_space = df_selected_space.dropna()
        df_selected_space = to_date(df_selected_space)

        return df_selected_space

    def select_features(self, df) -> pd.DataFrame:
        print("Ozellik cikarimi yapiliyor...")
        df = df[self.feature_list]
        df = self.change_icao24(df)

        return df

    @staticmethod
    def change_icao24(df) -> pd.DataFrame:
        print("ICAO24 numaraları integer değerine çevriliyor...")
        icao24 = df.loc[:, 'icao24']
        uniqueValues = icao24.unique()
        for i in range(0, len(uniqueValues)):
            df = df.replace([uniqueValues[i]], i + 1)

        return df

    @staticmethod
    def create_constant_attack(df) -> pd.DataFrame:
        # print("create constant attack")
        attack_values = [0.05, 0.0002, -10, 10, 0.2, -0.05]
        df_attack = df.copy()

        for i in range(0, len(df)):
            df_attack["lat"].iloc[i] = df["lat"].iloc[i] + attack_values[0]
            df_attack["lon"].iloc[i] = df["lat"].iloc[i] + attack_values[1]
            df_attack["velocity"].iloc[i] = df["velocity"].iloc[i] + attack_values[2]
            df_attack["heading"].iloc[i] = df["heading"].iloc[i] + attack_values[3]
            df_attack["baroaltitude"].iloc[i] = df["baroaltitude"].iloc[i]  # + attack_values[4]
            df_attack["geoaltitude"].iloc[i] = df["geoaltitude"].iloc[i]  # + attack_values[5]

        return df_attack

    @staticmethod
    def create_velocity_attack(df) -> pd.DataFrame:
        # print("create velocity attack")
        df_attack = df.copy()
        velocity_value = 0
        for i in range(0, len(df)):
            df_attack["velocity"].iloc[i] = df["velocity"].iloc[i] - velocity_value
            velocity_value = velocity_value + 0.0005

        return df_attack

    # @staticmethod
    def create_random_attack(self, df) -> pd.DataFrame:
        # print("create random attack")
        df_attack = df.copy()
        attack_values = [[0.05, -0.0002, -10],
                         [0.01, -0.003, 20],
                         [0.07, -0.05, -50],
                         [0.004, -0.06, 50],
                         [0.00009, -0.007, -25],
                         [0.04, -0.008, 29]]

        for i in range(0, len(df_attack)):
            index = rnd.randint(0, 5)
            df_attack["lat"].iloc[i] = df["lat"].iloc[i] + attack_values[index][0]
            df_attack["lon"].iloc[i] = df["lon"].iloc[i] + attack_values[index][1]
            df_attack["velocity"].iloc[i] = df["velocity"].iloc[i] + attack_values[index][2]

        return df_attack

    def create_attack(self, df) -> pd.DataFrame:
        print("Saldiri verileri uretiliyor")
        if self.attack_type == "constant":
            print(" Saldiri tipi: constant")
            df = self.create_constant_attack(df)
        elif self.attack_type == "random":
            print(" Saldiri tipi: random")
            df = self.create_random_attack(df)
        elif self.attack_type == "velocity":
            print(" Saldiri tipi: constant velocity")
            df = self.create_velocity_attack(df)
        elif self.attack_type == "all":
            print(" Saldiri tipi: all")
            df_velocity = self.create_velocity_attack(df)
            df_velocity = df_velocity.head(df_velocity.shape[0]//3)
            df_velocity["label"] = 0
            df_random = self.create_random_attack(df)
            df_random = df_random.head(df_random.shape[0]//3)
            df_random["label"] = 0
            df_constant = self.create_constant_attack(df)
            df_constant = df_constant.head(df_constant.shape[0]//3)
            df_constant["label"] = 0

            frames = [df_velocity, df_random, df_constant]
            df = pd.concat(frames)

        else:
            print("Yanlis saldiri tipi girilmistir!")
            return 0

        return df

    @staticmethod
    def merge_real_attack_data(df, df_attack) -> pd.DataFrame:
        df_attack = df_attack.head(1000)
        frames = [df, df_attack]
        df_res = pd.concat(frames)
        df_res = df_res.dropna()

        return df_res

    def run(self) -> None:
        self.label = 1
        df = self.get_dataset()
        self.label = 0
        df_attack = self.get_dataset()

        df_result = self.merge_real_attack_data(df, df_attack)

        print("Sonuc dataframe'e yaziliyor...")
        # res_path = self.f_path + "resultData\\result_dataframe.csv"
        res_path = "C:\\Users\\nursa\\Documents\\Nursah\\TİK 3\\Flight\\VsCode\\data\\resultData\\result_dataframe.csv"
        df_result.to_csv(res_path, index=False)
        

# f_path = "C:\\Users\\ncevik\\Desktop\\TİK2\\Flight\\VsCode\\data\\"
# mn = CreateDataset(f_path)
# mn.run()
