# Handin 1
Samuel Grund   
Zhang Leyi 202101646    
## PART I: Logistic Regression
### Code   
![LR_Acc](https://github.com/KirakiraZLY/ml-g39/blob/main/handin1/Img/LR_ISE.png?raw=true)
![LR_ISE](https://github.com/samgrund/ml-g39/blob/main/handin1/Img/LR_ISE.png?raw=true)

```python
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
        Yk = one_in_k_encoding(y, self.num_classes) # may help - otherwise you may remove it
        ### YOUR CODE HERE
        grad = -1 / y.shape[0] * X.transpose() @ (Yk - softmax(X@W))
        cost = -1 / y.shape[0] * np.sum(Yk * np.log(softmax(X @ W)))
        ### END CODE
        # print("cost, grad:",cost, grad)
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
        if W is None: W = np.zeros((X.shape[1], self.num_classes))
        history = []
        ### YOUR CODE HERE
        for i in range(0,epochs):
            sample_list = list(np.random.permutation(Y.shape[0]))
            X = X[sample_list]
            Y = Y[sample_list]

            for start in range(0,Y.shape[0],batch_size):
                stop = start + batch_size
                X_batch = X[start:stop]
                Y_batch = Y[start:stop]
                cost, grad = self.cost_grad(X_batch, Y_batch, W)
                W = W - lr * grad
            cost, _ = self.cost_grad(X, Y, W)
            print('Epoch: %d, Cost: %f' % (i, cost))
            history.append(cost)
        ### END CODE
        self.W = W
        self.history = history
```
### Theory
1. What is the running time of your mini-batch gradient descent algorithm?

The parameters:

n: number of training samples
d: dimensionality of training samples
epochs: number of epochs run
mini_batch_size: batch_size for mini_batch_gradient_descent
Write both the time to compute the cost and the gradient for log_cost You can assume that multiplying an  𝑎×𝑏  matrix with a  𝑏×𝑐  matrix takes  𝑂(𝑎𝑏𝑐)  time.

2. Sanity Check:
Assume you are using Logistic Regression for classifying images of cats and dogs. What happens if we randomly permute the pixels in each image (with the same permutation) before we train the classifier? Will we get a classifier that is better, worse, or the same than if we used the raw data? Give a short explanation (at most three sentences). HINT: The location of pixels relative to each other seem to hold some kind of information. Does a random permutation of all pixels position affect this locality? Does the model we use exploit pixel locality?

3. Linearly Separable Data:
If the data is linearly separable, what happens to weights when we implement logistic regression with gradient descent? That is, how do the weights that minimize the negative log likelihood look like? You may assume that we have full precision (that is, ignore floating point errors) and we can run gradient descent as long as we want (i.e. what happens with the weights in the limit).

Give a short explanation for your answer. You may include math if it helps (at most 5 lines).