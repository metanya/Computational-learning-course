import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn


def create_new_data(X_digits, y_digits, new_class, feature1, feature2):
    X_data = np.copy(X_digits)
    y_data = np.copy(y_digits)
    size = len(X_data)-1
    for i in range(size, -1, -1):
        if y_data[i] != new_class[0] and y_data[i] != new_class[1] and y_data[i] != new_class[2]:
            X_data = np.delete(X_data, i, axis=0)
            y_data = np.delete(y_data, i, axis=0)

    feature1 = X_data[:, feature1].reshape(-1, 1)
    feature2 = X_data[:, feature2].reshape(-1, 1)
    X_data = np.concatenate((feature1, feature2), axis=1)
    return X_data, y_data


def choose_3_class_and_feature():
    feature1 = 26
    feature2 = 43
    class1 = 0
    class2 = 1
    class3 = 2
    new_class = list()
    new_class.append(class1)
    new_class.append(class2)
    new_class.append(class3)
    new_class = np.array(new_class)
    return new_class, feature1, feature2


def get_W_j(x_i, averages_vector, corr_matrix, z, j, k):
    # P(Xi | Zj) * P(Zj)
    wj = conditional_probability_x_condition_z(x_i, averages_vector[j], corr_matrix[j]) * z[j]
    x_i_prob = np.array([])

    for i in range(k):# calculate P(Xi)
        x_i_prob = np.append(x_i_prob, z[i] * conditional_probability_x_condition_z(x_i, averages_vector[i], corr_matrix[i]))
    wj /= np.sum(x_i_prob)
    return wj


def conditional_probability_x_condition_z(x_i, averages_vector_j, corr_matrix_j):
    x_minus_av = x_i - averages_vector_j
    exponent = np.exp(-0.5 * np.dot(np.dot(x_minus_av.transpose(), np.linalg.inv(corr_matrix_j)), x_minus_av))
    conditional = ((2 * np.pi) ** (-len(x_i) / 2)) * (np.linalg.det(corr_matrix_j) ** (-0.5))
    conditional = conditional * exponent
    return conditional


def predict_y_pred(X_data, averages_vector, corr_matrix, m, k):
    prob = list()
    for i in range(m):
        x_i_prob = np.array([])
        for j in range(k): # For every class calculate if Xi belong to that class
            x_i_prob = np.append(x_i_prob, conditional_probability_x_condition_z(X_data[i], averages_vector[j], corr_matrix[j]))
        prob.append(x_i_prob)
    prob = np.array(prob)

    y_pred = np.array([])
    for i in range(len(prob)): # take the maximum probability that Xi belong to that class
        y_pred = np.append(y_pred, np.argmax(prob[i]))
    return y_pred


def GMM(X_data, m, k):
    iterations = 15
    z = np.array([])
    for i in range(k):
        z = np.append(z, 1/k)

    split_X_data = np.array_split(X_data, k)

    averages_vector = list()
    corr_matrix = list()
    for x in split_X_data:
        averages_vector.append(np.mean(x, axis=0))
        corr_matrix.append(np.cov(x.transpose()))
    averages_vector = np.array(averages_vector)
    corr_matrix = np.array(corr_matrix)

    for iteration in range(iterations):
        w = np.zeros((m, k)) #The probability of example if class j p(J|Xi) =

        # Expectation step:
        for i in range(m):
            for j in range(k):
                w[i][j] = get_W_j(X_data[i], averages_vector, corr_matrix, z, j, k)
        sum_w = np.sum(w, axis=0) # Calculate the sum of every column j of W

        #####################################################

        # Maximization step:

        for j in range(k):
            z[j] = sum_w[j] / m

        averages_vector = np.zeros((k, len(X_data[0])))

        for j in range(k):
            for i in range(m):
                averages_vector[j] += w[i][j]*X_data[i]
            averages_vector[j] /= sum_w[j]

        for j in range(k):
            corr_matrix[j] = np.cov(X_data.transpose(), aweights=(w[:, j]), ddof=0) / sum_w[j]

    # exit the loopךםךת

    y_pred = predict_y_pred(X_data, averages_vector, corr_matrix, m, k)
    return y_pred


def main_func():
    X_digits, y_digits = load_digits(return_X_y=True)
    new_class, feature1, feature2 = choose_3_class_and_feature()
    X_data, y_data = create_new_data(X_digits, y_digits, new_class, feature1, feature2)
    X_data = StandardScaler().fit_transform(X_data)
    m = len(y_data)     # amount of examples
    K_num_of_class = len(new_class)
    y_pred = GMM(X_data, m, K_num_of_class)

    Accuracy = accuracy_score(y_data, y_pred) * 100
    Accuracy = round(Accuracy, 3)

    conf = confusion_matrix(y_data, y_pred)
    sn.heatmap(conf, annot=True, cmap='coolwarm', fmt='g')
    plt.yticks(rotation=0)
    plt.ylabel('Actual', labelpad=22).set_rotation(0)
    plt.xlabel('Predicted')
    plt.title("GMM - " + str(Accuracy) + "%")
    plt.show()


if __name__ == "__main__":
    main_func()
