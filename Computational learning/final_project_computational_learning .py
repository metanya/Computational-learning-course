import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import RFE

TEST_SIZE = 0.33
VAL_TEST_SIZE = 0.1


def split_train_validation_test(x, y):
    """
    split the data x and y to 3 parts: train, validation and test
    :param x: all samples of the data
    :param y: class for each sample
    :return: all the split data of the x and y
    """
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(x, y, test_size=VAL_TEST_SIZE,
                                                                              random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation,
                                                                    test_size=TEST_SIZE, random_state=42)
    return X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_validation, y_train_validation


def print_confusion_matrix(y_test, y_pred, title):
    """
    get the Actual y and the Predicted y, compare between them and print the confusion matrix
    :param y_test: get the Actual y - for each sample which class he belong too
    :param y_pred:get the Predicted y - for each sample which class we predict he belong too
    :param title: name of the algorithm
    """
    conf = confusion_matrix(y_test, y_pred)
    Accuracy = accuracy_score(y_test, y_pred) * 100
    Accuracy = round(Accuracy, 3)
    sn.heatmap(conf, annot=True, cmap='coolwarm', fmt='g')
    plt.yticks(rotation=0)
    plt.ylabel('Actual', labelpad=22).set_rotation(0)
    plt.xlabel('Predicted')
    plt.title(str(title) + "\nAccuracy = " + str(Accuracy) + "%")
    plt.show()


def logistic_regression_algorithm(x, y):
    """
    run logistic regression algorithm with no parameters
    :param x: all samples of the data
    :param y: class for each sample
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    # Fit is the learning for the model
    lr = LogisticRegression(multi_class='multinomial', tol=0.00001, max_iter=len(x_train)).fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print_confusion_matrix(y_test, y_pred, "Logistic Regression Algorithm")


def one_vs_all(x, y, get_class):
    """
    run one vas all algorithm, each class compare to all the other classes and build model for each class
    :param x: all samples of the data
    :param y: class for each sample
    :param get_class: all the classes (1-7)
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
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
        get_max_class = max(predicts, key=predicts.get)
        y_pred = np.append(y_pred, get_max_class)

    # create the confusion matrix according to y_test and y_pred
    print_confusion_matrix(y_test, y_pred, "One vs All")


def one_vs_one(x, y, get_class):
    """
    run one vas one algorithm, each class compare to different class, for each class compare separately to all the classes
    :param x: all samples of the data
    :param y: class for each sample
    :param get_class: all the classes (1-7)
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
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
            for i in range(size - 1, -1, -1):
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
    print_confusion_matrix(y_test, y_pred, "One vs One")


def regulation_logistic_regression_algorithm(x, y):
    """
    run logistic regression algorithm with change of C parameter to build the best model
    :param x: all samples of the data
    :param y: class for each sample
    :return: the best model and the best C of that model
    """
    # split the data to train validation and test
    x_train, x_validation, x_test, y_train, y_validation, y_test, x_train_validation, y_train_validation = split_train_validation_test(
        x, y)
    iteration = 10
    C = 0.0001  # starter C for the loop of the learning loop
    best_C = C
    best_Accuracy = 0
    # loop to find the C for the best model
    for i in range(iteration):
        # build model with different C every iteration
        lr = LogisticRegression(C=C, penalty='l2', solver='lbfgs', multi_class='multinomial', tol=0.00001,
                                max_iter=len(x_train) * 5).fit(x_train, y_train)
        # check the model and get predict y on the validation
        y_pred = lr.predict(x_validation)
        # find the Accuracy for the specific C
        Accuracy = accuracy_score(y_validation, y_pred) * 100
        Accuracy = round(Accuracy, 3)
        # if the new C give better Accuracy from the last C then change the best_C
        if best_Accuracy < Accuracy:
            best_Accuracy = Accuracy
            best_C = C
        C = C * 10

    # after finding the best C we build the best model
    lr = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs', multi_class='multinomial', tol=0.00001,
                            max_iter=len(x_train_validation) * 5).fit(x_train_validation, y_train_validation)
    y_pred = lr.predict(x_test)
    # print the confusion matrix for the best model to compare to the model with the 2 best features
    print_confusion_matrix(y_test, y_pred, "Regulation best model")
    return lr, C


def RFE_3_best_features(x, y, lr, C, num_of_features=3):
    """
    run RFE algorithm to find the best features in the data
    :param x: all samples of the data
    :param y: class for each sample
    :param lr: the best model we found in the regulation
    :param C: the best C we found in the regulation
    :param num_of_features: number of the best features we want to found (3)
    """
    # split the data to train and test - no need for validation because we change the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    # run RFE algorithm using sklearn library
    rfe = RFE(lr, n_features_to_select=num_of_features)
    lr = rfe.fit(x_train, y_train)

    # get an array of the best features - the best features will mark as true and other ad false
    best_features = lr.support_
    features_index = (np.ones(num_of_features) * -1).astype('int')
    k = 0
    # find the features index
    for i in range(len(best_features)):
        if best_features[i]:
            features_index[k] = i
            k += 1
    # build new data that contain only the best features
    features = np.array([])
    for i in range(num_of_features):
        if i == 0:
            features = x[:, features_index[i]].reshape(-1, 1).copy()
        else:
            feature = x[:, features_index[i]].reshape(-1, 1)
            features = np.concatenate((features, feature), axis=1)
    # split the new data to train and test
    x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=TEST_SIZE, random_state=42)
    # l2 and lbfgs is calculate the norma 2
    lr = LogisticRegression(C=C, penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=len(x_train)).fit(
        x_train, y_train)
    y_pred = lr.predict(x_test)
    print_confusion_matrix(y_test, y_pred, "RFE - 3 best features: " + str(features_index))


def RFE_3_worst_features(x, y, lr, C, num_of_features=3):
    """
    run RFE algorithm to find the best features in the data using that we will found the worst features
    :param x: all samples of the data
    :param y: class for each sample
    :param lr: the best model we found in the regulation
    :param C: the best C we found in the regulation
    :param num_of_features: number of the worst features we want to found (3)
    """
    # split the data to train and test - no need for validation because we change the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    # run RFE algorithm using sklearn library
    # we search for the number of features - num_of_features
    rfe = RFE(lr, n_features_to_select=len(x[0]) - num_of_features)
    lr = rfe.fit(x_train, y_train)

    # get an array of the best features - the best features will mark as true and other ad false
    # because we search for the number of features - num_of_features the number of false features will be num_of_features
    best_features = lr.support_
    features_index = (np.ones(num_of_features) * -1).astype('int')
    k = 0
    # find the features index
    for i in range(len(best_features)):
        if not best_features[i]:
            features_index[k] = i
            k += 1
    # build new data that contain only the best features
    features = np.array([])
    for i in range(num_of_features):
        if i == 0:
            features = x[:, features_index[i]].reshape(-1, 1).copy()
        else:
            feature = x[:, features_index[i]].reshape(-1, 1)
            features = np.concatenate((features, feature), axis=1)
    # split the new data to train and test
    x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=TEST_SIZE, random_state=42)
    lr = LogisticRegression(C=C, penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=len(x_train)).fit(
        x_train, y_train)
    y_pred = lr.predict(x_test)
    print_confusion_matrix(y_test, y_pred, "RFE - 3 worst features: " + str(features_index))


def RFE_n_best_features(x, y, lr, C, num_of_features):
    """
    run RFE algorithm to find the best features in the data
    we will run this function to find every iteration n best features
    :param x: all samples of the data
    :param y: class for each sample
    :param lr: the best model we found in the regulation
    :param C: the best C we found in the regulation
    :param num_of_features:
    :return: return the Accuracy of the model we build
    """
    # split the data to train and test - no need for validation because we change the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    # run RFE algorithm using sklearn library
    rfe = RFE(lr, n_features_to_select=num_of_features)
    lr = rfe.fit(x_train, y_train)

    # get an array of the best features - the best features will mark as true and other ad false
    best_features = lr.support_
    features_index = (np.ones(num_of_features) * -1).astype('int')
    k = 0
    # find the features index
    for i in range(len(best_features)):
        if best_features[i]:
            features_index[k] = i
            k += 1
    # build new data that contain only the best features
    features = np.array([])
    for i in range(num_of_features):
        if i == 0:
            features = x[:, features_index[i]].reshape(-1, 1).copy()
        else:
            feature = x[:, features_index[i]].reshape(-1, 1)
            features = np.concatenate((features, feature), axis=1)
    # split the new data to train and test
    x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=TEST_SIZE, random_state=42)
    lr = LogisticRegression(C=C, penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=len(x_train)).fit(
        x_train, y_train)
    y_pred = lr.predict(x_test)
    # calculate the model Accuracy
    Accuracy = accuracy_score(y_test, y_pred) * 100
    Accuracy = round(Accuracy, 3)
    return Accuracy


def RFE_grow_of_best_features(x, y, lr, C):
    """
    run in iterations to find model of n features
    each iteration grow in 2 features
    we start to find model of 2 features then 4, 6, 8...
    :param x: all samples of the data
    :param y: class for each sample
    :param lr: the best model we found in the regulation
    :param C: the best C we found in the regulation
    """
    accuracy_list = np.array([])
    num_of_features_list = np.array([])
    # run the RFE for iteration*2
    for i in range(2, len(x[0]) + 2, 2):
        accuracy = RFE_n_best_features(x, y, lr, C, i)
        accuracy_list = np.append(accuracy_list, accuracy)
        num_of_features_list = np.append(num_of_features_list, i)

    # print the chart
    plt.plot(num_of_features_list, accuracy_list, marker='o')
    for a, b in zip(num_of_features_list, accuracy_list):
        plt.text(a, b, str(b))
    plt.title('Best Features')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Best Features')
    plt.show()
    plt.close()


def main_func():
    data = pd.read_csv('../zoo.csv')  # get the data from he csv file
    x = data.drop('animal_name', 1)  # get all the data and remove the y column
    x = np.array(x.drop('class_type', 1))  # get all the data and remove the y column
    y = np.array(data['class_type'])  # get the actual Y
    x = StandardScaler().fit_transform(x)  # normalize the data
    classes = np.unique(y)  # get all the classes

    logistic_regression_algorithm(x, y)

    one_vs_all(x, y, classes)
    one_vs_one(x, y, classes)

    lr, C = regulation_logistic_regression_algorithm(x, y)
    RFE_3_best_features(x, y, lr, C)
    RFE_3_worst_features(x, y, lr, C)

    RFE_grow_of_best_features(x, y, lr, C)


if __name__ == "__main__":
    main_func()
