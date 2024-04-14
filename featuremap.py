import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        n_examples = np.shape(X)[0]
        dim = np.shape(X)[1]
        if self.theta == None:
            self.theta = np.zeros((dim, 1))

        self.theta=np.linalg.solve((X.T)@X,(X.T)@y)    #normal eqn #doing this to avoid working with inverse
        #rank=np.linalg.matrix_rank((X.T)@X)
        #print("rank:",rank)

        return self.theta

        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n_examples = np.shape(X)[0]
        #dim = np.shape(X)[1]

        #n_examples,dim=X.shape
        phi=np.zeros((n_examples,k+1))    #output dimensions given above as a hint
        i=0
        for i in np.arange(n_examples):
            pow=0
            for pow in np.arange(k+1):
                phi[pow]=X[i]**pow

        return phi
        #print("Phi",phi)

        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***

        #sin=np.zeros(n_examples,k+2)

        n_examples = np.shape(X)[0]
        dim = np.shape(X)[1]
        phi = np.zeros(n_examples, k + 2)  # output dimensions given above as a hint
        i = 0
        for i in np.arange(n_examples):
            pow = 0
            for pow in np.arange(k + 1):
                phi[i, pow] = X[i, 1] ** pow
            phi[i,k+2]=np.sin(X[i,1])

        return phi
        #print("Phi", phi)

        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        n_examples = np.shape(X)[0]
        dim = np.shape(X)[1]
        pred=np.zeros((n_examples,1))
        i=0
        for i in np.arange(n_examples):
            pred[i]=pred[i]+(self.theta.T)@X[i,1]
        return pred

        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***

        clf=LinearModel()
        phi=clf.create_poly(k,train_x)
        if sine==True:
            phi=clf.create_sin(k,train_x)
        clf.fit(phi,train_y)
        plot_phi=clf.create_poly(k,plot_x)
        if sine==True:
            plot_phi=clf.create_sin(k,plot_x)
        plot_y=clf.predict(plot_phi)

        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***

    # Load training set
    # Train the logistic regression classifier
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    clf=LinearModel()               #calling the class
    phi=clf.create_poly(3,y_train)      #k=3 for degree 3 polynomial
    clf.fit(phi,y_train)            #notice the phi instead of x_train

    #Plotting for 4a #using the code from run_exp
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    plot_phi=clf.create_poly(3,plot_x)      #using different variable plot_phi
    plot_y=clf.predict(plot_phi)            #prediction with plot_phi

    plt.ylim(-2, 2)
    plt.plot(plot_x[:, 1], plot_y, label='for k=3')
    plt.legend()
    plt.savefig('4b.png')
    plt.clf()

    run_exp(train_path, filename='plot4b.png')
    run_exp(train_path, sine=True, filename='plot4c.png')
    run_exp(small_path, filename='plot4d.png')

    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
