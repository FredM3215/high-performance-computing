"""M3C 2018 Homework 2 Frederica Melbourne CID: 01068192"""

import numpy as np
import matplotlib.pyplot as plt
from w1 import nmodel as nm #assumes that hw2_dev.f90 has been compiled with: f2py -c hw2_dev.f90 -m m1
# May also use scipy, scikit-learn, and time modules as needed
import scipy.optimize
import time

def read_data(tsize=60000):
    """Read in image and label data from data.csv.
    The full image data is stored in a 784 x 70000 matrix, X
    and the corresponding labels are stored in a 70000 element array, y.
    The final 70000-tsize images and labels are stored in X_test and y_test, respectively.
    X,y,X_test, and y_test are all returned by the function.
    You are not required to use this function.
    """
    print("Reading data...") #may take 1-2 minutes
    Data=np.loadtxt('data.csv',delimiter=',')
    Data =Data.T
    X,y = Data[:-1,:]/255.,Data[-1,:].astype(int)%2 #rescale the image, convert the labels to 0s and 1s (For even and odd integers)
    Data = None

    # Extract testing data
    X_test = X[:,tsize:]
    y_test = y[tsize:]
    print("processed dataset")
    return X,y,X_test,y_test
#----------------------------

def snm_test(X,y,X_test,y_test,omethod,input=(None)):
    """Train single neuron model with input images and labels (i.e. use data in X and y), then compute and return testing error in test_error
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed

    One advantage of using this Python-Fortran approach is that when the Fortran code is compiled, the compiler analyzes it and optimizes it
    where possible. Therefore we are able to use the large do loop over the d datasets without losing efficiency. In Python this large for loop
    would take much more time to run.

    Another advantage of developing the code in Fortran is that Fortran has the necessary functions (dot product and exponential, for example)
    built in, whereas if you were writing in Python you would need to import another module (numpy) anyway to use these.
    """
    n = X.shape[0]
    fvec0 = np.random.randn(n+1) #initial fitting parameters

    if input == (None):
       dt = 20000
    else:
        dt = input


    #Add code to train SNM and evaluate testing test_error

    nm.data_init(n, dt)
    nm.nm_x = X[:,0:dt]
    nm.nm_y = y[0:dt]

    if omethod == 1:
        fvec_f=scipy.optimize.minimize(nm.snmodel, fvec0, args=(dt), method='L-BFGS-B', jac=True, options={'disp':True}).x

    elif omethod == 2:
        fvec_f=nm.sgd(fvec0,n,0,dt,0.1)

    a = np.zeros(10000)
    for i in range(10000):
        z = np.dot(fvec_f[:n], X_test[:,i]) + fvec_f[n]
        a[i] = 1/(1+np.exp(-z))

    round = np.rint(a)

    diffs = y_test - round

    ncorrect = sum((diffs==0))

    test_error = 1- (ncorrect/10000)

    return fvec_f,test_error
#--------------------------------------------

def nnm_test(X,y,X_test,y_test,m,omethod,input=(None)):
    """Train neural network model with input images and labels (i.e. use data in X and y), then compute and return testing error (in test_error)
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    m: number of neurons in inner layer
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed
    """
    n = X.shape[0]
    fvec0 = np.random.randn(m*(n+2)+1) #initial fitting parameters

    if input == (None):
       dt = 20000
    else:
       dt = input
    #Add code to train NNM and evaluate testing error, test_error

    nm.data_init(n, dt)
    nm.nm_x = X[:,0:dt]
    nm.nm_y = y[0:dt]

    #scipy method
    if omethod == 1:
        fvec_f=scipy.optimize.minimize(nm.nnmodel, fvec0, args=(n, m, dt), method='L-BFGS-B', jac=True, options={'disp':True}).x

    #sgd method
    elif omethod == 2:
        fvec_f=nm.sgd(fvec0,n,m,dt,0.1)

    ### code to calculate test error ###

    #get parameters
    winners =np.zeros((m,n))
    for i in range(n):
        winners[:,i] = fvec_f[i*m:(i+1)*m]
    bs = np.transpose([fvec_f[n*m:(n+1)*m],]*10000)
    b = ([fvec_f[m*(n+2)],]*10000)

    #calculate inner activations for each image
    zs = np.matmul(winners,X_test) + bs
    avec = np.reciprocal(1+np.exp(-zs))

    #caluclate outer activation for each image
    zhats = np.matmul(fvec_f[(n+1)*m:(n+2)*m], avec) + b
    ahats = np.reciprocal(1+np.exp(-zhats))

    #get test error
    round = np.rint(ahats)
    diffs = y_test - round
    ncorrect = sum((diffs==0))
    test_error = 1- (ncorrect/10000)

    return fvec_f, test_error
#--------------------------------------------

def nm_analyze(X,y,X_test,y_test,m,input=(None)):
    """ Analyze performance of single neuron and neural network models
    on even/odd image classification problem
    Add input variables and modify return statement as needed.
    Should be called from
    name==main section below

    Analysis of Graphs/Trends:

    The first graph shows the effect that increasing the number of internal neurons has on the test error when using the neural network model.
    It seems that the SGD method is more accurate when using three neurons or less, but after this the two methods give the same level of accuracy.
    We can also see that both methods show a decreasing error when the number of neurons are increased. However, most of the improvement occurs up to
    using 4 internal neurons; after this it appears that little accuracy is gained by adding more neurons (as the graph levels out).

    The second graph illustrates that the SGD method for the 20000 training images is much slower than the scipy minimize method. Also, as the number of
    internal neurons is increased, the SGD becomes slower at a rate of 25s per neuron added. In contrast, the time taken for the scipy minimize method
    stays roughly constant as more neurons are added. Together, these two graphs suggest that the neural network model using 4 internal neurons and the
    scipy minimize method may be the most effective, for 20000 training images.

    The last two graphs show how the training size d affects the accuracy and runtime of the snmodel and the nmodel with three inner neurons. We can see
    that the snmodel is considerably faster than the nmodel, but also more inaccurate (which becomes more apparent as d increases). Also, it is clear that
    for the snmodel, there is little difference between the accuracy and runtime of the SGD and Scipy methods, whereas for the nmodel the difference is
    significant- the SGD method is slower and the runtime grows more rapidly as d increases. It appears that the SGD method is more accurate for small
    training size but that the two methods have roughly similar levels of accuracy for d >10000.

    Finally, it is interesting to note that the runtimes of both methods for both the single and multiple neuron models grow linearly with respect to d. """

    #fig1, fig2
    error1min = np.zeros(m)
    t1min = np.zeros(m)
    error1sgd = np.zeros(m)
    t1sgd = np.zeros(m)
    for i in range(1,m+1):
        t1=time.time()
        error1min[i-1] = nnm_test(X, y, X_test, y_test, i, 1)[1]
        t2=time.time()
        error1sgd[i-1] = nnm_test(X, y, X_test, y_test, i, 2)[1]
        t3=time.time()
        t1min[i-1]=t2-t1
        t1sgd[i-1]=t3-t2

    #fig 1- increasing no. neurons effect on error
    plt.xlabel('m = no. internal neurons')
    plt.ylabel('test error')
    plt.title('no. internal neurons vs test_error for nnm_test')
    plt.plot(range(1,m+1), error1min, 'r', label='minimize')
    plt.plot(range(1,m+1), error1sgd, 'r--', label='SGD')
    plt.legend()
    plt.show()

    #fig 2- increasing no. neurons effect on run-time
    plt.figure()
    plt.xlabel('m= no. internal neurons')
    plt.ylabel('time taken to run (s)')
    plt.title('no. internal neurons vs time taken for nnm_test')
    plt.plot(range(1,m+1), t1min, 'r', label='minimize')
    plt.plot(range(1,m+1), t1sgd, 'r--', label='SGD')
    plt.legend()
    plt.show()

    # fig3, fig4
    error2smin = np.zeros(len(input))
    t2smin = np.zeros(len(input))
    error2ssgd = np.zeros (len(input))
    t2ssgd = np.zeros(len(input))
    error2nmin = np.zeros(len(input))
    t2nmin = np.zeros(len(input))
    error2nsgd = np.zeros(len(input))
    t2nsgd = np.zeros(len(input))
    for i in range(len(input)):
        t21=time.time()
        error2smin[i] = snm_test(X, y, X_test, y_test, 1, input = (input[i]))[1]
        t22=time.time()
        error2ssgd[i] = snm_test(X, y, X_test, y_test, 2, input = (input[i]))[1]
        t23=time.time()
        error2nmin[i] = nnm_test(X, y, X_test, y_test, 3, 1, input = (input[i]))[1]
        t24=time.time()
        error2nsgd[i] = nnm_test(X, y, X_test, y_test, 3, 2, input = (input[i]))[1]
        t25=time.time()
        t2smin[i]=t22-t21
        t2ssgd[i]=t23-t22
        t2nmin[i]=t24-t23
        t2nsgd[i]=t25-t24

    #fig3 - increasing number of training images effect on error
    plt.figure()
    plt.xlabel('d = no. training images')
    plt.ylabel('test error')
    plt.title('d vs test_error for snm_test and nnm_test')
    plt.plot(list(input), error2smin, 'b', label='snmodel, minimize' )
    plt.plot(list(input), error2ssgd, 'b--', label='snmodel, sgd')
    plt.plot(list(input), error2nmin, 'r', label='nnmodel with m=3, minimize')
    plt.plot(list(input), error2nsgd, 'r--', label='nnmodel with m=3, sgd')
    plt.legend()
    plt.show()

    #fig4 - increasing number of training images effect on run-time
    plt.figure()
    plt.xlabel('d = no.training images')
    plt.ylabel('time taken to run (s)')
    plt.title('d vs time taken to run (s) for snm_test and nnm_test')
    plt.plot(list(input), t2smin, 'b', label ='snmodel, minimize')
    plt.plot(list(input), t2ssgd, 'b--', label='snmodel, sgd')
    plt.plot(list(input), t2nmin, 'r', label='nnmodel, minimize')
    plt.plot(list(input), t2nsgd, 'r--', label='nnmodel, sgd')
    plt.legend()
    plt.show()

    return None
#--------------------------------------------

def display_image(X):
    """Displays image corresponding to input array of image data"""
    n2 = X.size
    n = np.sqrt(n2).astype(int) #Input array X is assumed to correspond to an n x n image matrix, M
    M = X.reshape(n,n)
    plt.figure()
    plt.imshow(M)
    return None
#--------------------------------------------
#--------------------------------------------


if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    X,y,X_test,y_test = read_data(tsize=60000)
    output = nm_analyze(X,y,X_test,y_test,10,input=(2000,5000,10000,15000,20000,30000))
