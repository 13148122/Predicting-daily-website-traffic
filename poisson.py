import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Fit a Poisson Regression model
    clf = PoissonRegression()
    theta = clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_predict = clf.predict(x_eval)
    np.savetxt(save_path, y_predict)
    #print(x_eval.shape)
    #print(y_predict)
    #print(theta.shape)

    # plotting the decision boundary on top of the validation set
    util.plot(x_eval, y_eval, theta, save_path+'plot.jpg', correction=1.0)

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        n_examples = np.shape(x)[0]
        dim = np.shape(x)[1]
        if self.theta == None:
            self.theta = np.zeros((dim,1))
        alpha=self.step_size                #alpha value given
        loss_fn=np.zeros((dim,1))             #not taking the 0th column
        iter=0
        while (iter>self.max_iter) and (norm_diff>self.eps):        #limiting the norm difference in parameter to 10^-5
            i=0
            for i in np.arange(n_examples):
                j=0
                for j in np.arange(dim):
                    loss_fn[j]=loss_fn[j]+(y[i]-np.exp(self.theta.T@x[i,:]))*x[i,j]


            theta_upd=self.theta+alpha*loss_fn
            norm_diff=np.sum(np.absolute(theta_upd-self.theta))     #limiting the norm difference in parameter to 10^-5
            self.theta=theta_upd            #updating the theta for the next iteration
            iter+=1
            loss_fn = np.zeros((dim, 1))
            print("Iter No",iter)

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***

        n_examples = np.shape(x)[0]
        dim = np.shape(x)[1]
        y = np.zeros(n_examples) #initialize vector y to store predictions
        for i in range(n_examples):
            xp = x[i, :].reshape([dim, 1])
            y[i] = 1 / (1 + np.exp(-np.transpose(self.theta) @ xp))     #since Poisson Distribution belongs to the exponential family
        return y

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
