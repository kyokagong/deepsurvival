import sys
import argparse
from data_loader import DatasetLoader
from settings import DATASET_ROOT_DIR

def preprocess_data():
    save_path = './new_features.csv'
    survival_data_save_path = './new_survival_data.csv'
    data_loader = DatasetLoader()
    features = data_loader.load_cell_features(DATASET_ROOT_DIR, save_path=save_path)
    print(features.shape)
    print(data_loader.patient_id_list)
    survival_data = data_loader.load_survial_data(DATASET_ROOT_DIR, save_path=survival_data_save_path)
    print(survival_data.shape)


def train_model():
    from dsm import datasets, DeepSurvivalMachines
    import numpy as np
    from sksurv.metrics import concordance_index_ipcw, brier_score

    survival_data = np.loadtxt('./new_survival_data.csv', delimiter=',')
    features = np.loadtxt('./new_features.csv', delimiter=',')

    x = features
    t = survival_data[:, 0]
    e = survival_data[:, 1]

    times = np.quantile(t[e == 1], [0.25, 0.5, 0.75]).tolist()

    cv_folds = 2
    folds = list(range(cv_folds))*10000
    folds = np.array(folds[:len(x)])


    cis = []
    brs = []
    for fold in range(cv_folds):

        print("On Fold:", fold)

        x_train, t_train, e_train = x[folds != fold], t[folds != fold], e[folds != fold]
        x_test, t_test, e_test = x[folds == fold], t[folds == fold], e[folds == fold]
        print(x_train.shape)

        model = DeepSurvivalMachines(distribution='Weibull', layers=[100])
        model.fit(x_train, t_train, e_train, iters=10, learning_rate=1e-3, batch_size=10)

        et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                            dtype=[('e', bool), ('t', int)])

        et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                        dtype=[('e', bool), ('t', int)])

        out_risk = model.predict_risk(x_test, times)
        out_survival = model.predict_survival(x_test, times)

        cis_ = []
        for i in range(len(times)):
            cis_.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
        cis.append(cis_)

        brs.append(brier_score(et_train, et_test, out_survival, times)[1])

    print("Concordance Index:", np.mean(cis, axis=0))
    print("Brier Score:", np.mean(brs, axis=0))


if __name__ == '__main__':
    args = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", help="func", default='check')

    args = parser.parse_args()
    if args.func == 'preprocess_data':
        preprocess_data()
    elif args.func == 'train_model':
        train_model()
