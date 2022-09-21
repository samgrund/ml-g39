import numpy as np
from h1_util import numerical_grad_check


def softmax(X):
    """ 
    Compute the softmax of each row of an input matrix (2D numpy array).

    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 

    More precisely this is what you must do.

    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    # YOUR CODE HERE
    for row in range(X.shape[0]):  # for each row
        max = np.amax(X[row, :])  # find max
        # compute logsum
        logsum = np.log(np.sum(np.exp(X[row, :] - max))) + max
        res[row, :] = np.exp(X[row, :] - logsum)  # compute softmax
    # END CODE
    return res


def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 

    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc


class SoftmaxClassifier():

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.W = None

    def cost_grad(self, X, y, W):
        """ 
        Compute the average negative log likelihood cost and the gradient under the softmax model 
        using data X, Y and weight matrix W.

        the functions np.log, np.nonzero, np.sum, np.dot (@), may come in handy
        Args:
           X: numpy array shape (n, d) float - the data each row is a data point
           y: numpy array shape (n, ) int - target values in 0,1,...,k-1
           W: numpy array shape (d x K) float - weight matrix
        Returns:
            totalcost: Average Negative Log Likelihood of w 
            gradient: The gradient of the average Negative Log Likelihood at w 
        """
        cost = np.nan
        grad = np.zeros(W.shape)*np.nan
        # may help - otherwise you may remove it
        Yk = one_in_k_encoding(y, self.num_classes)
        # YOUR CODE HERE
        # Formula from lecture using the one_in_k_encoding
        grad = -1/len(y) * X.transpose() @ (Yk - softmax(X @ W))
        # Sanity check: Do the gradient and W have the same shape?
        assert grad.shape == W.shape, 'gradient and W shape mismatch'
        # Formula from lecture
        cost = -1/len(y) * np.sum(Yk * np.log(softmax(X @ W)))
        # END CODE
        return cost, grad

    def fit(self, X, Y, W=None, lr=0.01, epochs=10, batch_size=16):
        """
        Run Mini-Batch Gradient Descent on data X,Y to minimize the in sample error (1/n)NLL for softmax regression.
        Printing the performance every epoch is a good idea to see if the algorithm is working

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
           W: numpy array shape (d x K)
           lr: scalar - initial learning rate
           batchsize: scalar - size of mini-batch
           epochs: scalar - number of iterations through the data to use

        Sets: 
           W: numpy array shape (d, K) learned weight vector matrix  W
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        if W is None:
            W = np.zeros((X.shape[1], self.num_classes))
        history = []
        # YOUR CODE HERE
        for epoch in range(epochs):
            # shuffle data
            # this is a list of random numbers from 0 to len(y)
            perm = np.random.permutation(len(Y))
            X = X[perm]  # select the rows of X in the order of the random numbers
            Y = Y[perm]  # select the rows of y in the order of the random numbers
            # For each batch
            for i in range(0, len(X), batch_size):
                batch_x = X[i:i + batch_size]
                batch_y = Y[i:i + batch_size]
                # compute cost and gradient
                cost, grad = self.cost_grad(batch_x, batch_y, W)
                # update weights
                W = W - lr * grad
            # save final cost of epoch on full training set
            cost, _ = self.cost_grad(X, Y, W)
            print('Epoch: %d, Cost: %f' % (epoch, cost))
            history.append(cost)
        # END CODE
        self.W = W
        self.history = history

    def score(self, X, Y):
        """ Compute accuracy of classifier on data X with labels Y

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
        Returns:
           out: float - mean accuracy
        """
        out = 0
        # YOUR CODE HERE
        out = np.sum(self.predict(X) == Y)/len(Y)
        # END CODE
        return out

    def predict(self, X):
        """ Compute classifier prediction on each data point in X 

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
        Returns
           out: np.array shape (n, ) - prediction on each data point (number in 0,1,..., num_classes-1)
        """
        out = None
        # YOUR CODE HERE
        model_predictions = np.dot(X, self.W)
        # get the index of the highest value in each row, this is the prediction.
        # axis=1 means we are looking at each row, not each column (axis 0)
        out = np.argmax(model_predictions, axis=1)
        assert out.shape == (X.shape[0],), 'prediction shape mismatch'
        # END CODE
        return out


def test_encoding():
    print('*'*10, 'test encoding')
    labels = np.array([0, 2, 1, 1])
    m = one_in_k_encoding(labels, 3)
    res = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    assert res.shape == m.shape, 'encoding shape mismatch'
    assert np.allclose(m, res), m - res
    print('Test Passed')


def test_softmax():
    print('Test softmax')
    X = np.zeros((3, 2))
    X[0, 0] = np.log(4)
    X[1, 1] = np.log(2)
    print('Input to Softmax: \n', X)
    sm = softmax(X)
    expected = np.array([[4.0/5.0, 1.0/5.0], [1.0/3.0, 2.0/3.0], [0.5, 0.5]])
    print('Result of softmax: \n', sm)
    assert np.allclose(
        expected, sm), 'Expected {0} - got {1}'.format(expected, sm)
    print('Test complete')


def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    w = np.ones((2, 3))
    y = np.array([0, 1, 2])
    scl = SoftmaxClassifier(num_classes=3)
    def f(z): return scl.cost_grad(X, y, W=z)
    numerical_grad_check(f, w)
    print('Test Success')


if __name__ == "__main__":
    test_encoding()
    test_softmax()
    test_grad()
