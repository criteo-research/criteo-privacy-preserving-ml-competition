import os
import numpy as np
from zipfile import ZipFile
from datetime import datetime
import urllib.request

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

X_TRAIN_FN = 'X_train.csv.gz'
Y_TRAIN_FN = 'y_train.csv.gz'
X_TEST_FN = 'X_test.csv.gz'
AGG_DATA_URL = 'http://go.criteo.net/criteo-privacy-ml-competition-data/aggregated-noisy-data-pairs.csv.gz'
AGG_DATA_FN = 'aggregated_noisy_data_pairs.csv.gz'
AGG_DATA_SINGLE_FN = 'aggregated_noisy_data_singles.csv.gz'

Y_HAT_CLICK_FN = 'y_hat_click.txt'
Y_HAT_SALE_FN = 'y_hat_sale.txt'


def download_additional_data_if_needed():
    if not os.path.exists(AGG_DATA_FN):
        print("downloading additional data...", end='')
        with urllib.request.urlopen(AGG_DATA_URL) as response, open(AGG_DATA_FN, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print('finished:', AGG_DATA_FN)


def load_data(click=True, sale=False, load_aggregated_data=True, load_aggregated_data_pairs=True):
    X_train = np.loadtxt(X_TRAIN_FN, skiprows=1, delimiter=',')
    y_train = np.loadtxt(Y_TRAIN_FN, skiprows=1, delimiter=',').astype(np.int8)
    if click and not sale:
        y_train = y_train[:, 0]
    elif sale and not click:
        y_train = y_train[:, 1]
    X_test = np.loadtxt(X_TEST_FN, skiprows=1, delimiter=',')
    Xy_agg_data = Xy_agg_data_singles = None
    if load_aggregated_data:
        download_additional_data_if_needed()
        Xy_agg_data_singles = np.loadtxt(AGG_DATA_SINGLE_FN, skiprows=1, delimiter=',')
        if load_aggregated_data_pairs:
            Xy_agg_data = np.loadtxt(AGG_DATA_FN, skiprows=1, delimiter=',')
    return X_train, y_train, Xy_agg_data_singles, Xy_agg_data, X_test


def create_submission(y_hat_click, y_hat_sale=None, filename: str = None, description: str = None):
    """Method to export your solution.

    The zip file
      - must contain at the root (not in a subdirectory) a file \"y_hat_click.txt\" and/or a file \"y_hat_sale.txt\"

    These files
      - shall contain individual predictions (a float in [0;1]), one per line
      - must be of the same length as 'X_test.csv.gz' (in number of lines)
    """
    np.savetxt(Y_HAT_CLICK_FN, y_hat_click, fmt='%1.6f')
    if y_hat_sale is not None:
        np.savetxt(Y_HAT_SALE_FN, y_hat_sale, fmt='%1.6f')
    if filename is None:
        filename = 'submission-%s.zip' % str(datetime.now()).replace(' ', '_').replace(':', '-')
    with ZipFile(filename, 'w') as zip:
        zip.write(Y_HAT_CLICK_FN)
        if y_hat_sale is not None:
            zip.write(Y_HAT_SALE_FN)
        if description is not None and len(description):
            zip.writestr('description', description)
    print('wrote', filename)


if __name__ == '__main__':

    # load data
    X_train, y_train, _, _, X_test = load_data(load_aggregated_data=False)
    print('data loaded')
    ## keep a validation set for offline evaluation
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=.8, random_state=0)

    # learn model
    clf = make_pipeline(OneHotEncoder(handle_unknown='ignore'),
                        LogisticRegression(solver='liblinear'))
    clf.fit(X_train, y_train)
    print('model learned')

    # simulate task locally
    print("train loss   : %.6f" % log_loss(y_train, clf.predict_proba(X_train)[:, 1]))
    print("baseline loss: %.6f" % log_loss(y_valid, np.ones(len(y_valid)) * y_train.mean()))
    print("valid loss   : %.6f" % log_loss(y_valid, clf.predict_proba(X_valid)[:, 1]))

    # predict on test data
    y_hat_click = clf.predict_proba(X_test)[:, 1]

    # create zip file to upload
    assert X_test.shape[0] == y_hat_click.shape[0], \
        "invalid prediction shape: %s expected %s" % (X_test.shape[0], y_hat_click.shape[0])
    create_submission(y_hat_click, description='my first submission')
