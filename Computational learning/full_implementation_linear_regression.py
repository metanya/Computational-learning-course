import numpy as np
from matplotlib import pyplot as plt


# section A
def normal_regulations_of_vector(vector):
    uj = np.average(vector)  # find Average Attribute of Uj
    stdj = np.std(vector)  # find Standard Deviation of Uj
    print("Average Attribute: " + str(uj) + ", Standard Deviation: " + str(stdj))
    for i in range(len(vector)):
        vector[i] = (vector[i] - uj) / stdj
    # check if after the normal regulations if the Average Attribute = 0 and Standard Deviation = 1
    uj = np.average(vector)  # find Average Attribute of Uj after the normal regulations
    stdj = np.std(vector)  # find Standard Deviation of Uj after normal the regulations
    print("Normal => Average Attribute: " + str(uj) + ", Standard Deviation: " + str(stdj) + "\n")
    # I used an error of 0.01 to check if the uj and stdj are close to 1 and 0
    epsilon = 0.001
    if not(((0-epsilon) < uj < (0+epsilon)) and ((1-epsilon) < stdj < (1+epsilon))):
        print("Error in normal regulations matrix\n")
        return


def normal_regulations_of_matrix(matrix):
    copy_matrix = np.copy(matrix)
    for i in range(len(copy_matrix[0])):
        vec = copy_matrix[:, i:i+1]
        normal_regulations_of_vector(vec)
    return copy_matrix


# section B
def get_h_theta(vector_theta, vector_x):
    hthetax = 0
    if len(vector_theta) == len(vector_x)+1:  # if there is not 1 in the start of vector X
        vector_x = np.concatenate(([1], vector_x), axis=0)
    if len(vector_theta) == len(vector_x):  # if there is 1 in the vector X
        hthetax = np.dot(vector_x, vector_theta)
    else:  # if one of the vector in longer from the other
        print("Error the size of vector X and theta are not fit to the calculation\n")
    return hthetax


# section C
def get_j_theta(vector_theta, matrix_x, matrix_y):
    jtheta = 0
    m = len(matrix_y)
    for i in range(m):
        hthetax = get_h_theta(vector_theta, matrix_x[i])
        jtheta += (hthetax - matrix_y[i])**2
    jtheta = jtheta / (2*m)
    return jtheta


# section D
def get_gradient_j_theta(vector_theta, matrix_x, matrix_y):
    matrix_x_transpose = matrix_x.transpose()
    gradientjtheta = np.matmul(matrix_x, vector_theta)
    gradientjtheta = np.subtract(gradientjtheta, matrix_y)
    gradientjtheta = np.matmul(matrix_x_transpose, gradientjtheta)
    gradientjtheta = gradientjtheta / len(matrix_x)
    return gradientjtheta


# section E
def run_gradient_descent(matrix_x, matrix_y, alpha, theta, iterations):
    k = 0
    j_theta_history = np.zeros(iterations)
    while k < iterations:
        gradient_j_theta = get_gradient_j_theta(theta, matrix_x, matrix_y)
        theta = theta - alpha * gradient_j_theta
        j_theta_history[k] = get_j_theta(theta, matrix_x, matrix_y)# the error size
        k += 1

    # print the graph of the theta
    plt.plot(j_theta_history)
    plt.title('Gradient Descent - Alpha = ' + str(alpha))
    plt.ylabel('J(θ)')
    plt.xlabel('Iterations')
    plt.show()
    plt.close()


# section F
def run_minibatch(matrix_x, matrix_y, alpha, theta, iterations):
    k = 0
    m = len(matrix_x)  # len of m 3047
    n = len(matrix_x[0])  # len of n 9
    t_group = 30
    n_size = int(m/t_group)
    j_theta_history = np.zeros(iterations)
    while k < iterations:
        for j in range(n):
            for i in range((k*n_size) % m, ((k+1) * (n_size-1)) % m):
                gradient = (matrix_x[i][j] * (get_h_theta(theta, matrix_x[i]) - matrix_y[i])) / n_size
                theta[j] = theta[j] - alpha * gradient
        j_theta_history[k] = get_j_theta(theta, matrix_x, matrix_y)
        k += 1

    # print the graph of the theta
    plt.plot(j_theta_history)
    plt.title('Minibatch - Alpha = ' + str(alpha))
    plt.ylabel('J(θ)')
    plt.xlabel('Iterations')
    plt.show()
    plt.close()


# section G
def run_momentum(matrix_x, matrix_y, alpha, theta, iterations):
    k = 0
    beta = 0.9
    j_theta_history = np.zeros(iterations)
    vector_v = np.array([])
    while k < iterations:
        gradient_j_theta = get_gradient_j_theta(theta, matrix_x, matrix_y)
        if k == 0:
            vector_v = alpha * gradient_j_theta
        else:
            vector_v = beta * vector_v + (alpha * gradient_j_theta)
        theta = theta - vector_v
        j_theta_history[k] = get_j_theta(theta, matrix_x, matrix_y)
        k += 1

    # print the graph of the theta
    plt.plot(j_theta_history)
    plt.title('Momentum - Alpha = ' + str(alpha))
    plt.ylabel('J(θ)')
    plt.xlabel('Iterations')
    plt.show()
    plt.close()


def main_func():
    data = np.genfromtxt('cancer_data.csv', delimiter=',')
    matrix_x = data[:, :len(data[0]) - 1]
    matrix_y = data[:, len(data[0]) - 1:len(data[0])]
    alpha = np.array([0.1, 0.01, 0.001])
    # create theta of 0.5
    theta = np.ones((len(matrix_x[0])+1, 1)) / 2
    # normalize the matrix X and vector y
    normalize_matrix_x = normal_regulations_of_matrix(matrix_x)
    normalize_matrix_y = normal_regulations_of_matrix(matrix_y)

    m = len(normalize_matrix_x)  # len of m 3047
    # add vector 1 to matrix X for the calculation mul with theta
    normalize_matrix_x = np.hstack((np.ones((m, 1)), normalize_matrix_x))

    iterations = 100  # do 100 iterations in the algorithm
    # run the gradient descent algorithm
    run_gradient_descent(normalize_matrix_x, normalize_matrix_y, alpha[0], theta, iterations)
    run_gradient_descent(normalize_matrix_x, normalize_matrix_y, alpha[1], theta, iterations)
    run_gradient_descent(normalize_matrix_x, normalize_matrix_y, alpha[2], theta, iterations)
    # run the minibatch algorithm
    # run_minibatch(normalize_matrix_x, normalize_matrix_y, alpha[0], theta, iterations)
    # run_minibatch(normalize_matrix_x, normalize_matrix_y, alpha[1], theta, iterations)
    # run_minibatch(normalize_matrix_x, normalize_matrix_y, alpha[2], theta, iterations)
    # run the momentum algorithm
    run_momentum(normalize_matrix_x, normalize_matrix_y, alpha[0], theta, iterations)
    run_momentum(normalize_matrix_x, normalize_matrix_y, alpha[1], theta, iterations)
    run_momentum(normalize_matrix_x, normalize_matrix_y, alpha[2], theta, iterations)


if __name__ == "__main__":
    main_func()
