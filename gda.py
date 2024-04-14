import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***

    # Train a GDA classifier
    clf = GDA()
    theta = clf.fit(x_train, y_train)

    # evaluate validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    y_predict = clf.predict(x_eval)
    x_eval=util.add_intercept(x_eval)

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, y_predict)

    # plot decision boundary on top of validation set
    util.plot(x_eval, y_eval, theta, save_path+ 'plot1e.jpg', correction=1.0)

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Write theta in terms of the parameters

        n_examples = np.shape(x)[0]     #gives me the rows of x
        dim = np.shape(x)[1]            #gives me the columns of x

        # in case a prediction (theta) was already given
        if self.theta == None:
            self.theta = np.zeros(1+dim)

        # Find phi, mu_0, mu_1, and sigma

        # initialize sums to zero
        phi = 0
        mu0_num = np.zeros(dim)
        mu0_den = 0
        mu1_num = np.zeros([dim,1])
        mu1_den = 0
        sig = 0

        # finding phi
        for i in range(n_examples):
            xp = x[i, :].reshape([dim, 1])
            if y[i] == 1:
                phi = phi + 1                 #summation of indicator function when y=1

        #finding mu_0 and mu_1
            if y[i] == 1:
                mu1_num = mu1_num + xp        #summation of (indicator function)*x(i) when y=1
                mu1_den = mu1_den + 1         #summation of indicator function when y=1
            else:
                mu0_num = mu0_num + xp        #summation of (indicator function)*x(i) when y=0
                mu0_den = mu0_den + 1         #summation of indicator function when y=0

        # perform final division phi
        phi = phi / n_examples

        # perform final divisions for mu_0 and mu_1
        mu_0 = mu0_num/mu0_den
        mu_1 = mu1_num/mu1_den

        # finding sigma
        mu_i = np.zeros(x.shape)
        mu_i[y == 0,:] = mu_0[:,0]            #finding the ith element of mu_0 for indicator at y=0
        mu_i[y == 1,:] = mu_1[:,0]            #finding the ith element of mu_1 for indicator at y=1

        sigma = np.transpose(x-mu_i)@(x-mu_i)

        # perform final division for sigma
        sigma = sigma/n_examples

        # Write theta in terms of the parameters

        theta = np.linalg.inv(sigma)@(mu_1-mu_0)
        th0 = (np.transpose(mu_0)@np.linalg.inv(sigma)@mu_0) - (np.transpose(mu_1)@np.linalg.inv(sigma)@mu_1)

        th0 = (.5*th0) - np.log((1-phi)/phi)

        self.theta[0] = th0[0,0]

        print("THETA", theta)
        self.theta[1:3] = theta[:,0]
        print(self.theta)
        return self.theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        n_examples = np.shape(x)[0]     #gives me the rows of x
        dim = np.shape(x)[1]            #gives me the columns of x
        y = np.zeros(n_examples)        #initializing y (a vector) to store predictions
        for i in range(n_examples):
            xp = x[i, :].reshape([dim, 1])
            y_i = 1 / (1 + np.exp(-np.transpose(self.theta[1:]) @ xp)+self.theta[0])
            y[i] = y_i
        return y

        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
