import numpy as np
from matplotlib import pyplot as plt

data_size = 20
mu, sigma = 0, 0.04
_min, _max = 0, 1
degrees = [5, 10, 14]
train_size = round(data_size * 0.75)
test_size = data_size - train_size

def create_data(num):
    eps = np.random.normal(mu, sigma, (num,))
    x = np.random.uniform(_min, _max, (num,))
    y = np.sin(2 * np.pi * x) + eps
    
    return x, y

def least_squares_method(x, y):

    return np.linalg.inv(x.T @ x) @ x.T @ y

def mse(x, y):
    return np.mean((y - x) ** 2) 

def kfold_validation(train_input, train_target, fold):
    idx = np.arange(len(train_input))
    losses = np.zeros((fold,))

    for i in range(fold):
        select = idx % fold == i
        val_x = train_input[select]
        train_x = train_input[~select]
        val_y = train_target[select]
        train_y = train_target[~select]
        try:
            weight = least_squares_method(train_x, train_y)
        except:
            continue
        predict = val_x @ weight
        loss = mse(predict, val_y)
        losses[i] = loss

    return losses.mean()
    
def loo_validation(train_input, train_target):
    idx = np.arange(len(train_input))
    losses = np.zeros((len(train_input),))
    
    for i in idx:
        select = idx == i
        val_x = train_input[select]
        train_x = train_input[~select]
        val_y = train_target[select]
        train_y = train_target[~select]
        try:
            weight = least_squares_method(train_x, train_y)
        except:
            continue
        predict = val_x @ weight
        loss = mse(predict, val_y)
        losses[i] = loss

    return losses.mean()

def draw(ax, _input, target, model_x, model_y, degree):
    ax.plot(model_x, model_y, label=f"Degree{degree}")
    ax.scatter(_input, target, marker='o', edgecolors='g', color=[])
    ax.legend()


if __name__ == '__main__':
    
    x, y = create_data(train_size + test_size)
    train, test = x[:train_size], x[train_size:]
    train_target = y[:train_size]
    test_target = y[train_size:]
    show = np.arange(_min, _max + .01, .01)
    plt.figure(figsize=(8, 6))
    train_ax = plt.subplot(121)
    test_ax = plt.subplot(122)
    train_ax.set_xlim(_min, _max)
    test_ax.set_xlim(_min, _max)

    for degree in degrees:
       
        train_input = np.ones((train_size, degree + 1))
        test_input = np.ones((test_size, degree + 1))
        show_input = np.ones((show.shape[0], degree + 1))

        for d in range(1, degree+1):
            show_input[:, d] = show ** d
            train_input[:, d] = train ** d
            test_input[:, d] = test ** d
    
        weight = least_squares_method(train_input, train_target)
        model_x = show_input[:, 1]
        model_y = show_input @ weight

        kfold_loss = kfold_validation(train_input, train_target, 5)
        loo_loss = loo_validation(train_input, train_target)

        ## plot train
        
        train_ax.set_title("Train")
        train_predict = train_input @ weight
        train_loss = mse(train_predict, train_target)
        draw(train_ax, train_input[:, 1], train_target, model_x, model_y, degree)
        
        ## plot test
        
        test_ax.set_title("Test")
        test_predict = test_input @ weight
        test_loss = mse(test_predict, test_target)
        draw(test_ax, test_input[:, 1], test_target, model_x, model_y, degree)

        
        print ("\nDegree " + str(degree))
        print ("Train Loss: {:.6f}".format(train_loss))
        print ("Kfold Loss: {:.6f}".format(kfold_loss))
        print ("Loo Loss: {:.6f}".format(loo_loss))
        print ("Test Loss: {:.6f}".format(test_loss))
    
    plt.savefig("c.png")