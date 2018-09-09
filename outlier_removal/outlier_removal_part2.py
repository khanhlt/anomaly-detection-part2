import numpy as np

re = np.loadtxt('tsne_result.txt', dtype=float)
name = np.loadtxt('image_name.txt', dtype=str)

outliers = []

# outliers is the tsne's element with
# 1.9 < x < 3 and -0.5 < y < 1
for i in range(len(re)):
    if ((1.95 < re[i][0] and re[i][0] < 3 and -0.5 < re[i][1] and re[i][1] < 1)):
        outliers.append(name[i])
print(np.asarray(outliers).shape)
print(outliers)
print(re.shape)
print(name.shape)