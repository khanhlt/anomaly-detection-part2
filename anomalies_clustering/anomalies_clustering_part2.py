import numpy as np

re = np.loadtxt('tsne_result_outliers.txt', dtype=float)
name = np.loadtxt('image_name_outliers.txt', dtype=str)

g1 = []
g2 = []
g3 = []


for i in range(len(re)):
    if ((-1 < re[i][0] and re[i][0] < 2 and -1.5 < re[i][1] and re[i][1] < 0)):
        g1.append(name[i])

for i in range(len(re)):
    if ((-1.5 < re[i][0] and re[i][0] < 0 and 1 < re[i][1] and re[i][1] < 2)):
        g2.append(name[i])

g3 = [x for x in name if x not in g1 and x not in g2]
print(name.shape)
print(g1)
print(g2)
print(g3)
print(np.asarray(g1).shape)
print(np.asarray(g2).shape)
print(np.asarray(g3).shape)
