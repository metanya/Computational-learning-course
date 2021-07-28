import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn


def split_train_validation_test(X, y):
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation,
                                                                    test_size=0.33, random_state=42)
    return X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation


def print_confusion_matrix(y_test, y_pred, title):
    conf = confusion_matrix(y_test, y_pred)
    Accuracy = accuracy_score(y_test, y_pred) * 100
    Accuracy = round(Accuracy, 3)
    sn.heatmap(conf, annot=True, cmap='coolwarm')
    plt.yticks(rotation=0)
    plt.ylabel('Actual', labelpad=22).set_rotation(0)
    plt.xlabel('Predicted')
    plt.title("Logistic Regression Lasso- " + str(Accuracy) + "%")
    plt.title("Logistic Regression " + str(title) + " - " + str(Accuracy) + "%")
    plt.show()


def lasso_selection(X_digits, y_digits, get_class):
    # split the data to train validation and test
    X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation = split_train_validation_test(
        X_digits, y_digits)
    C = 0.00001  # starter C for the loop of the learning loop
    best_C = C
    best_Accuracy = 0
    # loop to find the C for the best model
    for i in range(10):
        lr = LogisticRegression(C=C, penalty='l1', solver='saga', multi_class='multinomial', tol=0.001,
                                max_iter=len(X_train)).fit(X_train, y_train)  # build the model
        y_pred = lr.predict(X_validation)
        # find the Accuracy for the specific C
        Accuracy = accuracy_score(y_validation, y_pred) * 100
        Accuracy = round(Accuracy, 3)
        print("C: " + str(C) + " - Accuracy = " + str(Accuracy) + "%")
        # if the new C is give better Accuracy then change it to the
        if best_Accuracy < Accuracy:
            best_Accuracy = Accuracy
            best_C = C
        C = C * 10

    # after finding the best C we build the best model
    lr = LogisticRegression(C=best_C, penalty='l1', solver='saga', multi_class='multinomial', tol=0.001,
                            max_iter=len(X_train)).fit(X_train_validation, y_train_validation)
    y_pred = lr.predict(X_test)
    # print the confusion matrix for the best model to compare to the model with the 2 best features
    print_confusion_matrix(y_test, y_pred, "Lasso")

    # find the most important coefficient
    temp_coef = lr.coef_
    num_of_data = np.zeros(64)
    temp_list = list(y_digits)
    num_of_features = np.zeros((len(get_class)))
    # count how many examples are in each class
    for i in get_class:
        num_of_features[i] = temp_list.count(i)

    # count haw many data is reset to zero
    for i in range(len(temp_coef)):
        for k in range(len(temp_coef[i])):
            if temp_coef[i][k] != 0:
                num_of_data[k] += num_of_features[i]

    # find the 2 max columns that there is the most data
    max1 = max(num_of_data)
    max_arr = np.array([])
    for i in range(len(num_of_data)):
        if num_of_data[i] == max1:
            max_arr = np.append(max_arr, i)

    if len(max_arr) < 2:
        num_of_data[max_arr[0]] = 0
        max2 = max(num_of_data)
        for i in range(len(num_of_data)):
            if num_of_data[i] == max2:
                max_arr = np.append(max_arr, i)

    max_accuracy = 0
    index1 = 0
    index2 = 0
    # find the 2 best features that give the best Accuracy from all the max columns
    for i in range(len(max_arr)):
        for j in range(len(max_arr)):
            if j > i:
                # create data to check is Accuracy
                feature1 = X_digits[:, int(max_arr[i])].reshape(-1, 1)
                feature2 = X_digits[:, int(max_arr[j])].reshape(-1, 1)
                features = np.concatenate((feature1, feature2), axis=1)
                # split the data to train validation and test
                X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation = split_train_validation_test(
                    features, y_digits)
                lr = LogisticRegression(C=best_C, penalty='l1', solver='saga', multi_class='multinomial', tol=0.001,
                                        max_iter=len(X_train)).fit(X_train_validation, y_train_validation)
                y_pred = lr.predict(X_test)
                # find the Accuracy for the 2 best features
                Accuracy = accuracy_score(y_test, y_pred) * 100
                Accuracy = round(Accuracy, 3)
                if max_accuracy < Accuracy:
                    max_accuracy = Accuracy
                    index1 = int(max_arr[i])
                    index2 = int(max_arr[j])

    # build the data of the 2 best features
    feature1 = X_digits[:, index1].reshape(-1, 1)
    feature2 = X_digits[:, index2].reshape(-1, 1)
    features = np.concatenate((feature1, feature2), axis=1)

    X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation = split_train_validation_test(
        features, y_digits)
    lr = LogisticRegression(C=best_C, penalty='l1', solver='saga', multi_class='multinomial', tol=0.001,
                            max_iter=len(X_train)).fit(X_train_validation, y_train_validation)
    y_pred = lr.predict(X_test)
    print_confusion_matrix(y_test, y_pred, "Lasso 2 best features")
    print("Lasso 2 best features:")
    print(features)


def greedy_forward_selection(X_digits, y_digits):
    # split the data to train validation and test
    X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation = split_train_validation_test(
        X_digits, y_digits)
    C = 0.00001  # starter C for the loop of the learning loop
    best_C = C
    best_Accuracy = 0
    # loop to find the C for the best model
    for i in range(10):
        # train on split data
        lr = LogisticRegression(C=C, penalty='l2', solver='lbfgs', multi_class='multinomial', tol=0.001,
                                max_iter=len(X_train)).fit(X_train, y_train)
        y_pred = lr.predict(X_validation)
        # find the Accuracy for the specific C
        Accuracy = accuracy_score(y_validation, y_pred) * 100
        Accuracy = round(Accuracy, 3)
        print("C: " + str(C) + " - Accuracy = " + str(Accuracy) + "%")
        # if the new C is give better Accuracy then change it to the
        if best_Accuracy < Accuracy:
            best_Accuracy = Accuracy
            best_C = C
        C = C * 10

    # after finding the best C we build the best model
    lr = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs', multi_class='multinomial', tol=0.001,
                            max_iter=len(X_train)).fit(X_train_validation, y_train_validation)
    y_pred = lr.predict(X_test)
    # print the confusion matrix for the best model to compare to the model with the 2 best features
    print_confusion_matrix(y_test, y_pred, "Greedy Best model")

    best_features = np.array([])  # create array to save the index of the best features
    index = 0
    best_Accuracy = 0
    # start the first loop to find the first best feature that give the best Accuracy
    # start from the first feature to the last one
    for i in range(len(X_digits[0])):
        feature = X_digits[:, i]
        F = feature.reshape(-1, 1)
        # split the data to train validation and test
        X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation = split_train_validation_test(
            F, y_digits)
        lr = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs', multi_class='multinomial', tol=0.001,
                                max_iter=len(X_train)).fit(X_train, y_train)
        y_pred = lr.predict(X_validation)
        # find the Accuracy for the specific C
        Accuracy = accuracy_score(y_validation, y_pred) * 100
        Accuracy = round(Accuracy, 3)
        if best_Accuracy < Accuracy:
            best_Accuracy = Accuracy
            index = i

    best_features = np.append(best_features, index)  # save the first best feature
    # feature = X_digits[:, int(best_features[0])]
    # F = feature.reshape(-1, 1)
    index = 0
    best_Accuracy = 0
    # start fuse the features with the best feature we found
    for i in range(len(X_digits[0])):
        if i != int(best_features[0]):
            feature1 = X_digits[:, int(best_features[0])].reshape(-1, 1)
            feature2 = X_digits[:, i].reshape(-1, 1)
            F = np.concatenate((feature1, feature2), axis=1)
            X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation = split_train_validation_test(
                F, y_digits)
            lr = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs', multi_class='multinomial', tol=0.001,
                                    max_iter=len(X_train)).fit(X_train, y_train)
            y_pred = lr.predict(X_validation)
            # find the Accuracy for the specific C
            Accuracy = accuracy_score(y_validation, y_pred) * 100
            Accuracy = round(Accuracy, 3)
            # if the new C is give better Accuracy then change it to the
            if best_Accuracy < Accuracy:
                best_Accuracy = Accuracy
                index = i
    # finding the second feature and save it
    best_features = np.append(best_features, index)
    feature1 = X_digits[:, int(best_features[0])].reshape(-1, 1)
    feature2 = X_digits[:, int(best_features[1])].reshape(-1, 1)
    # create data that contain only the best 2 features
    F = np.concatenate((feature1, feature2), axis=1)
    # split the data to train validation and test
    X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation = split_train_validation_test(
        F, y_digits)
    lr = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs', multi_class='multinomial', tol=0.001,
                            max_iter=len(X_train)).fit(X_train_validation, y_train_validation)
    y_pred = lr.predict(X_test)
    # print the confusion matrix for the 2 best features to compare to the best model
    print_confusion_matrix(y_test, y_pred, "Greedy forward 2 best features")
    print("Greedy 2 best features:")
    print(F)


def main_func():
    X_digits, y_digits = load_digits(return_X_y=True)
    X_digits = StandardScaler().fit_transform(X_digits)
    get_class = np.unique(y_digits)  # get the 10 class
    print("--------------------Start Lasso:--------------------")
    lasso_selection(X_digits, y_digits, get_class)
    print("--------------------Start Greedy:--------------------")
    greedy_forward_selection(X_digits, y_digits)


if __name__ == "__main__":
    main_func()
