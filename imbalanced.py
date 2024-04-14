import numpy as np
import util
import sys
from random import random

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times

    # Train the logistic regression classifier
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    clf = LogisticRegression()
    theta = clf.fit(x_train, y_train)

    # Evaluating the validation dataset
    x_eval, y_eval = util.load_dataset(validation_path, add_intercept=True)
    y_predict = clf.predict(x_eval)

    # Using np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, y_predict)
    print(x_eval.shape)
    print("y_predict",y_predict)
    print(theta.shape)

    # plotting the decision boundary on top of the validation set
    util.plot(x_eval, y_eval, theta, save_path + 'plot5b.jpg')

    #plot accuracies
    Accuracy(y_predict,y_eval)

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # define the number of examples (variables)
        n_examples = np.shape(x)[0]     #gives me the rows of x
        dim = np.shape(x)[1]            #gives me the columns of x

        # in case a prediction (theta) was already given
        if self.theta == None:
            self.theta = np.zeros(dim)

        count=0                             #counting the iteration number
        norm_diff=1000                      #initialize difference
        while (count<=self.max_iter) and (norm_diff>self.eps):
            # defining hessian
            hess = np.zeros([dim,dim])      #Hessian is square matrix
            h = 0
            thetaTx = 0
            for i in range(n_examples):
                for k in range(dim):
                    thetaTx = thetaTx + self.theta[k].T * (x[i,k])  #sum of thetaTx
                sig = np.divide(1, 1 + np.exp(-1 * thetaTx))        #defining the sigmoid function g(thetaTx) as sig
                #print("x",x)
                xp=x[i,:].reshape([dim, 1])
                #print("xp",xp)

                h = h + (((sig*(1-sig))*xp)@np.transpose(xp))           #defining hessian without 1/n
                thetaTx = 0
            hess = h/n_examples + .0000001 #defining the actual hessian- adding a small number prevents singularity (if at all)

            print("Hessian:",hess)
            print("Hessian size:", hess.shape)

            # defining gradient
            grad = 0
            for i in range(n_examples):
                for k in range(dim):
                    thetaTx = thetaTx + self.theta[k].T * (x[i,k])  #sum of thetaTx
                g = np.divide(1, 1 + np.exp(-1 * thetaTx))
                xp = x[i,:].reshape([dim, 1])
                thetaTx=0
                grad = grad - ((y[i]-g)*xp)
            grad = grad/n_examples
            print("Gradient:", grad)
            print("Gradient size:",grad.shape)

            self.theta = self.theta.reshape([dim,1])
            theta_new = self.theta - (np.linalg.inv(hess)@grad)
            norm_diff=np.linalg.norm(theta_new-self.theta, ord=1)
            self.theta=theta_new

            print("Theta:", self.theta)

            count=count+1 #update iteration counter
            print(count)

        return self.theta

    # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        n_examples = np.shape(x)[0]
        dim = np.shape(x)[1]
        y = np.zeros(n_examples) #initialize vector to store predictions
        for i in range(n_examples):
            xp = x[i, :].reshape([dim, 1])
            y[i] = 1 / (1 + np.exp(-np.transpose(self.theta) @ xp))
        return y

def Accuracy(y_predict, y_eval):

    n_examples, dim = y_predict.shape
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    i = 0
    for i in np.arange(n_examples):
        if y_predict[i] > 0.5:
            if y_eval[i] == 1:
                TP += 1
            if y_eval[i] == 0:
                FP += 1
        if y_predict[i] < 0.5:
            if y_eval[i] == 1:
                FN += 1
            if y_eval[i] == 0:
                TN += 1
    A1 = TP / (TP + FN)
    A0 = TN / (TN + FP)
    A_bal = 0.5 * (A1 + A0)
    A = (TP + TN) / (TP + FP + TN + FN)
    print("A1:", A1)
    print("A0:", A0)
    print("A_bal:", A_bal)
    print("A:", A)
# *** END CODE HERE ***

    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
