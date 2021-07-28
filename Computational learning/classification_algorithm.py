import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sn


def one_vs_all(X, Y, get_class):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    lr_arr = {className: 0 for className in get_class}

    # train learn the data
    for class_name in get_class:
        y_train_copy = y_train.copy()
        # create class for 1 and 0
        for i in range(len(y_train_copy)):
            if class_name == y_train_copy[i]:
                y_train_copy[i] = 1
            else:
                y_train_copy[i] = 0
        # calculate the LogisticRegression for each class
        y_train_copy = y_train_copy.astype('int')
        lr = LogisticRegression().fit(x_train, y_train_copy)
        lr_arr[class_name] = lr

    # test start to check if the learn was good
    y_pred = np.array([])
    for row in x_test:
        predicts = {className: 0 for className in get_class}
        for class_name, lr in lr_arr.items():
            row = row.reshape(1, -1)
            # calculate the probability that the row belong to the class
            probability = lr.predict_proba(row)
            predicts[class_name] += probability[:, 1]
        # get the biggest probability to the class that the row belong to it
        get_max = max(predicts, key=predicts.get)
        y_pred = np.append(y_pred, get_max)

    # create the confusion matrix according to y_test and y_pred
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True, cmap='coolwarm')
    plt.yticks(rotation=360)
    plt.ylabel('Actual').set_rotation(0)
    # calculate the accuracy of our learning
    Accuracy = metrics.accuracy_score(y_test, y_pred)*100
    Accuracy = round(Accuracy, 3)
    plt.title("One vs All - Accuracy = " + str(Accuracy) + "%")
    plt.show()


def one_vs_one(X, Y, get_class):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    lr_matrix = []
    copy_class = get_class.copy()

    # train learn the data
    for first_class in get_class:
        copy_class = np.delete(copy_class, 0)
        for second_class in copy_class:
            x_t = x_train.copy()
            y_t = y_train.copy()
            size = len(y_train)
            # create data of first and second class only
            for i in range(size-1, -1, -1):
                if y_train[i] != first_class and y_train[i] != second_class:
                    x_t = np.delete(x_t, i, 0)
                    y_t = np.delete(y_t, i, 0)
            # create class for 1 and 0
            for i in range(len(y_t)):
                if y_t[i] == first_class:
                    y_t[i] = 1
                else:
                    y_t[i] = 0
            # calculate the LogisticRegression for each class
            y_t = y_t.astype('int')
            lr = LogisticRegression().fit(x_t, y_t)
            temp = [(first_class, second_class, lr)]
            lr_matrix += temp

    # test start to check if the learn was good
    y_pred = np.array([])
    for row in x_test:
        predicts = {className: 0 for className in get_class}
        for i in range(len(lr_matrix)):
            row = row.reshape(1, -1)
            # calculate the probability that the row belong to the class
            probability = lr_matrix[i][2].predict_proba(row)
            predicts[lr_matrix[i][0]] += probability[:, 1]
            predicts[lr_matrix[i][1]] += probability[:, 0]
        # get the biggest probability to the class that the row belong to it
        get_max = max(predicts, key=predicts.get)
        y_pred = np.append(y_pred, get_max)

    # create the confusion matrix according to y_test and y_pred
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True, cmap='coolwarm')
    plt.yticks(rotation=360)
    plt.ylabel('Actual').set_rotation(0)
    # calculate the accuracy of our learning
    Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    Accuracy = round(Accuracy, 3)
    plt.title("One vs One - Accuracy = " + str(Accuracy) + "%")
    plt.show()


def main_func():
    data = pd.read_excel("BreastTissue.xlsx", sheet_name="Data")    # get the data from he xlsx file
    data = data.drop('Case #', 1)   # delete the numbering line

    X = data.drop('Class', 1)
    X = StandardScaler().fit_transform(X)
    Y = np.array(data['Class'])
    get_class = np.unique(Y)  # get the 6 class

    one_vs_all(X, Y, get_class)
    one_vs_one(X, Y, get_class)


if __name__ == "__main__":
    main_func()
