from load_data import load_data
import numpy as np
import cv2
from sklearn.neighbors import LocalOutlierFactor


def extract_feature(data, vector_size=32):
    image_set = np.asarray(data)
    feature_set = []
    i = 1
    for image in image_set:
        try:
            alg = cv2.xfeatures2d.SURF_create()
            kps = alg.detect(image)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            kps, dsc = alg.compute(image, kps)
            dsc = dsc.flatten()
            needed_size = (vector_size * 32)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            print("Error: ", e)
            return None
        feature_set.append(dsc)
        print("features extraction: %d/%d" % (i, len(image_set)))
        i += 1
    return np.asarray(feature_set)


if __name__=="__main__":
    train, test, test_label, num_test_norm, num_test_anom = load_data()

    train = extract_feature(train)
    test = extract_feature(test)

    clf = LocalOutlierFactor(n_neighbors=10, leaf_size=50)
    clf.fit(train)
    pred_train = clf._predict(train)
    pred_test = clf._predict(test)
    print(pred_train)
    print(pred_test)


    tp, fp, tn, fn = 0., 0., 0., 0.
    for i in range(len(test)):
        if (test_label[i] == 1):
            if (pred_test[i] == -1):
                tp += 1
            elif (pred_test[i] == 1):
                fn += 1
        elif (test_label[i] == 0):
            if (pred_test[i] == -1):
                fp += 1
            elif (pred_test[i] == 1):
                tn += 1

    print('\nPrecesion: %.3f' % (tp / (tp + fp)))
    print('\nRecall: %.3f' % (tp / (tp + fn)))
    print('\nAccuracy: %.3f' % ((tp + tn) / (tp + tn + fn + fp)))
    print('\nAccuracy on anomaly: %d/%d' % (tp, num_test_anom))
    print('\nAccuracy on normal: %d/%d' % (tn, num_test_norm))