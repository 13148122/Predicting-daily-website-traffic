import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Train the logistic regression classifier
    clf = LogisticRegression()
    theta = clf.fit(x_train, y_train)

    # Evaluating the validation dataset
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    y_predict = clf.predict(x_eval)

    # Using np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, y_predict)
    print(x_eval.shape)
    print(y_predict)
    print(theta.shape)

    # plotting the decision boundary on top of the validation set
    util.plot(x_eval, y_eval, theta, save_path+'plot.jpg', correction=1.0)

    # *** END CODE HERE ***


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

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
