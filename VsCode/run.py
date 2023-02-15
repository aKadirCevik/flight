from createDataset import CreateDataset
from classify import Classify
from superLearner import Superlearner
import sys
import os


def run_test():
    sys.path.append(os.getcwd() + '\\data\\')
    path = sys.path[-1]

    print("Veri seti olusturuluyor.")
    # f_path = path
    f_path = 'C:\\Users\\nursa\\Documents\\Nursah\\TİK 3\\Flight\\VsCode\\data\\rawData\\states_2021-12-06-23.csv'
    cd = CreateDataset(f_path)
    cd.run()

    print("Base Modeller calistiriliyor.")
    f_path1 = path + 'resultData\\result_dataframe.csv'
    cls = Classify(f_path1)
    cls.run()

    print("Super Learner Model calistiriliyor.")
    sl = Superlearner(f_path)
    sl.run()


run_test()
