from my_first_submission import *

y_train = np.loadtxt(Y_TRAIN_FN, skiprows=1, delimiter=',').astype(np.int8)[:, 0]
y_bar = np.mean(y_train)

Xy_agg_data = np.loadtxt(AGG_DATA_SINGLE_FN, skiprows=1, delimiter=',')

# "learning"
d = dict()
for row in Xy_agg_data:
    # CTR by (feature x modality), regularized by global CTR and low-capped at 1e-9 =~ 0
    d[(int(row[1]), (int(row[0])))] = max(
        float(row[-2]+y_bar*100) / float(row[-3]+100),
        1e-9
    )

# offline "feature selection"
X_train = np.loadtxt(X_TRAIN_FN, skiprows=1, delimiter=',')
best_feature = None
best_loss = 1
for f in range(0, 19):
    # "prediction" = CTR for feature modality, or global CTR if modality unknown
    y_hat = [d.get((f, int(x[f])), y_bar) for x in X_train]
    loss = log_loss(y_train, y_hat)
    print("feature %2d -- valid loss   : %.6f" % (f, loss))
    if loss < best_loss:
        best_loss = loss
        best_feature = f
print("best aggregated feature:", best_feature, best_loss)

# prediction on test
X_test = np.loadtxt(X_TEST_FN, skiprows=1, delimiter=',')
y_hat = [d.get((best_feature, x[best_feature]), y_bar) for x in X_test]

create_submission(y_hat,
                  filename='submission-ctrdict-%s.zip' % str(datetime.now()).replace(' ', '_').replace(':', '-'),
                  description='agg data ctr dict')
