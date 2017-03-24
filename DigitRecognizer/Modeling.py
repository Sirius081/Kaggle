import ReadCsv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib

def pca(data, testData):
    print("pca...")
    pca = PCA(n_components=300)
    pca.fit(data)
    data = pca.transform(data)
    testData = pca.transform(testData)
    print(sum(pca.explained_variance_ratio_))
    return data, testData


def model(data, target):
    print("modeling...")
    # clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(data, target)
    return clf


def writeResult(result):
    print("write result...")
    out = open("result.csv", "w")
    out.write("ImageId,Label\r\n")
    count = 1
    for n in result:
        out.write("%d,%s\r\n" % (count, n))
        count += 1
    out.close()


print("reading data...")
data, target = ReadCsv.getTrainingData()
testData = ReadCsv.getTestingData()
ratio=0.99
train_data = data[0:ratio * len(data)]
test_data = testData
train_target = target[0:ratio * len(data)]

#test_target = target[ratio * len(data) + 1:]

train_data,  test_data = pca(train_data, test_data)
clf = model(train_data, train_target)

print("predicting...")
result = clf.predict(test_data)

print("writ ing result...")
writeResult(result)
# count = 1
# for i in range(0, len(result) - 1):
#     if result[i] == test_target[i]:
#         count += 1
# print(float(count)/len(result))
