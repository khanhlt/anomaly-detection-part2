import numpy as np
from sklearn import svm
from load_data import load_data_origin
from scipy.stats import entropy

norm, anom = load_data_origin()

pixel_entropy = np.array([entropy([x[i] for x in norm]) for i in np.arange(norm[0].shape[0])])
entropy_indices = np.array(pixel_entropy).argsort()


norm = np.array(norm)[:, entropy_indices[:512]]
anom = np.array(anom)[:, entropy_indices[:512]]
train = norm[:1000]
test_norm = norm[1000:1728]
test = np.concatenate((anom, test_norm), axis=0)
label = [1] * len(anom) + [0] * len(test_norm)

svm = svm.OneClassSVM(nu=0.05)
svm.fit(train)
pred_test = svm.predict(test)

tp, fp, tn, fn = 0., 0., 0., 0.
for i in range(len(test)):
    if (label[i] == 1):
        if (pred_test[i] == -1):
            tp += 1
        elif (pred_test[i] == 1):
            fn += 1
    elif (label[i] == 0):
        if (pred_test[i] == -1):
            fp += 1
        elif (pred_test[i] == 1):
            tn += 1

print('\nPrecesion: %.3f' % (tp / (tp + fp)))
print('\nRecall: %.3f' % (tp / (tp + fn)))
print('\nAccuracy: %.3f' % ((tp + tn) / (tp + tn + fn + fp)))
print('\nAccuracy on anomaly: %d/%d' % (tp, len(anom)))
print('\nAccuracy on normal: %d/%d' % (tn, len(norm)))
