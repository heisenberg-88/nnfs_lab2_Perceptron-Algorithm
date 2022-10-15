import numpy as np
import matplotlib.pyplot as plt


def step_func(z):
    if z>0:
        return 1.0
    else:
        return 0.0



def perceptron(x,y,learning_rate,epochs):
    showpoints(x, y)
    total_training_examples = x.shape[0]
    total_features = x.shape[1]
    theta = np.zeros((total_features + 1, 1))
    missed_list = []
    flag = False
    for epoch in range(epochs):
        print("epoch "+str(epoch)+"\n")
        missed = 0
        for index , x_data in enumerate(x):
            x_data = np.insert(x_data,0,1).reshape(-1,1)

            y_hat = step_func(np.dot(x_data.T, theta))

            if (np.squeeze(y_hat - y[index]) !=0) :
                theta += learning_rate * ((y[index] - y_hat) * x_data)
                missed+=1

        missed_list.append(missed)
        if(missed==0):
            print("There are no missed points.\n")
            print("early stopping...")
            flag = True

        plot_decision_boundary(x,y,theta)
        if(flag==True):
            break

    return theta,missed_list






def showpoints(x,y):
    fig = plt.figure(figsize=(10, 8))

    # points with 0 as label
    plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], 'r^')
    # points with 1 as label
    plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], 'bs')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Perceptron Algorithm")
    # block = False ensure that all figure windows are displayed and return immediately.
    plt.show()
    # plt.pause(1)
    # plt.close()

def plot_decision_boundary(x,y,theta):
    # X --> Inputs
    # theta --> parameters

    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = [min(x[:, 0]), max(x[:, 0])]
    m = -theta[1] / theta[2]
    c = -theta[0] / theta[2]
    x2 = m * x1 + c

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "r^")
    plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "bs")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Perceptron Algorithm")
    plt.plot(x1, x2, 'y-')
    # block = False ensure that all figure windows are displayed and return immediately.
    plt.show()
    # plt.pause(1)
    # plt.close()



