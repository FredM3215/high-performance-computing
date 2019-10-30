"""M3C 2018 Homework 3 Frederica Melbourne CID: 01068192
Contains five functions:
    plot_S: plots S matrix -- use if you like
    simulate2: Simulate tribal competition over m trials. Return: all s matrices at final time
        and fc at nt+1 times averaged across the m trials.
    performance: To be completed -- analyze and assess performance of python, fortran, and fortran+openmp simulation codes
    analyze: To be completed -- analyze influence of model parameter, g
    visualize: To be completed -- generate animation illustrating "interesting" tribal dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from y1 import tribes as tr #assumes that hw3_dev.f90 has been compiled with: f2py3 --f90flags='-fopenmp' -c hw3_dev.f90 -m f1 -lgomp
#May also use scipy and time modules as needed
import time
import matplotlib.animation as animation

def plot_S(S):
    """Simple function to create plot from input S matrix
    """
    ind_s0 = np.where(S==0) #C locations
    ind_s1 = np.where(S==1) #M locations
    plt.plot(ind_s0[1],ind_s0[0],'rs')
    plt.plot(ind_s1[1],ind_s1[0],'bs')
    plt.show()
    return None
#------------------


def simulate2(N,Nt,b,e,g,m):
    """Simulate m trials of C vs. M competition on N x N grid over
    Nt generations. b, e, and g are model parameters
    to be used in fitness calculations.
    Output: S: Status of each gridpoint at end of simulation, 0=M, 1=C
            fc_ave: fraction of villages which are C at all Nt+1 times
                    averaged over the m trials
    """

    #Set initial condition
    S  = np.ones((N,N,m),dtype=int) #Status of each gridpoint: 0=M, 1=C
    j = int((N-1)/2)
    S[j,j,:] = 0
    N2inv = 1./(N*N)

    fc_ave = np.zeros(Nt+1) #Fraction of points which are C
    fc_ave[0] = S.sum()

    #Initialize matrices
    NB = np.zeros((N,N,m),dtype=int) #Number of neighbors for each point
    NC = np.zeros((N,N,m),dtype=int) #Number of neighbors who are Cs
    S2 = np.zeros((N+2,N+2,m),dtype=int) #S + border of zeros
    F = np.zeros((N,N,m)) #Fitness matrix
    F2 = np.zeros((N+2,N+2,m)) #Fitness matrix + border of zeros
    A = np.ones((N,N,m)) #Fitness parameters, each of N^2 elements is 1 or b
    P = np.zeros((N,N,m)) #Probability matrix
    Pden = np.zeros((N,N,m))
    #---------------------

    #Calculate number of neighbors for each point
    NB[:,:,:] = 8
    NB[0,1:-1,:],NB[-1,1:-1,:],NB[1:-1,0,:],NB[1:-1,-1,:] = 5,5,5,5
    NB[0,0,:],NB[-1,-1,:],NB[0,-1,:],NB[-1,0,:] = 3,3,3,3
    NBinv = 1.0/NB
    #-------------

    #----Time marching-----
    for t in range(Nt):
        R = np.random.rand(N,N,m) #Random numbers used to update S every time step

        #Set up coefficients for fitness calculation
        A = np.ones((N,N,m))
        ind0 = np.where(S==0)
        A[ind0] = b

        #Add boundary of zeros to S
        S2[1:-1,1:-1,:] = S

        #Count number of C neighbors for each point
        NC = S2[:-2,:-2,:]+S2[:-2,1:-1,:]+S2[:-2,2:,:]+S2[1:-1,:-2,:] + S2[1:-1,2:,:] + S2[2:,:-2,:] + S2[2:,1:-1,:] + S2[2:,2:,:]

        #Calculate fitness matrix, F----
        F = NC*A
        F[ind0] = F[ind0] + (NB[ind0]-NC[ind0])*e
        F = F*NBinv
        #-----------

        #Calculate probability matrix, P-----
        F2[1:-1,1:-1,:]=F
        F2S2 = F2*S2
        #Total fitness of cooperators in community
        P = F2S2[:-2,:-2,:]+F2S2[:-2,1:-1,:]+F2S2[:-2,2:,:]+F2S2[1:-1,:-2,:] + F2S2[1:-1,1:-1,:] + F2S2[1:-1,2:,:] + F2S2[2:,:-2,:] + F2S2[2:,1:-1,:] + F2S2[2:,2:,:]

        #Total fitness of all members of community
        Pden = F2[:-2,:-2,:]+F2[:-2,1:-1,:]+F2[:-2,2:,:]+F2[1:-1,:-2,:] + F2[1:-1,1:-1,:] + F2[1:-1,2:,:] + F2[2:,:-2,:] + F2[2:,1:-1,:] + F2[2:,2:,:]

        P = (P/Pden)*g + 0.5*(1.0-g) #probability matrix
        #---------

        #Set new affiliations based on probability matrix and random numbers stored in R
        S[:,:,:] = 0
        S[R<=P] = 1

        fc_ave[t+1] = S.sum()
        #----Finish time marching-----

    fc_ave = fc_ave*N2inv/m

    return S,fc_ave
#------------------


def performance(input=(None),display=False):
    """Assess performance of simulate2, simulate2_f90, and simulate2_omp
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed

    Comments:

    The first graph shows how the performance of the Fortran/OpenMP implementation
    varies with no. trials M and grid size N. It contrasts the performance when using
    one and two threads. We can see that, as you would expect, runtime increases linearly
    with M (as the number of loops is being increased) and seems to increase proportional
    to N^2 (because, for example, doubling N leads to four times as many grid points).
    We can also see that using one thread (equivalent to a non-parallelized version) is notably
    slower than using two thread- this shows that parallelisation is successful in improving
    performance.

    The second graph explicitly shows this speedup of the Fortran/OpenMP implementation;
    calculated by fixing N and Nt and varying the number of trials, M, and
    using speedup = runtime with one thread / runtime with two threads. We can
    see that there is an average speedup of around 1.92 for all sample sizes, which
    doesn't appear to be affected by the number of trials. Thus there is a great
    improvement in performance by parallelising the function.

    Finally, we can compare the performance of the Fortran/OpenMP implementation to the pure
    Fortran and Python implementations. I chose to compare this for increasing M due to the found
    linear relationship, which could make trends easier to observe. We see that the relative
    performance is quite consistent- for all values of M that were tested, the Fortran version
    was about two fifths the speed of the Fortran/OpenMP version, and the Python version was much worse,
    at around a twentieth of the speed. This is because vectorisation is much more important in Python,
    whereas when Fortran code is compiled the code can generally be optimized. Thus the loop over time
    results in a much worse performance for the Python implementationself.

    The parallelized version is faster than both of the others because the tasks are being split up and
    then carried out simultaneously.

    """

    #set parameters
    tr.tr_b = 1.1
    tr.tr_e = 0.01
    tr.tr_g = 0.95

    #initialise arrays
    onethreadtimeM=np.zeros(7)
    onethreadtimeN=np.zeros(7)
    twothreadtimeM=np.zeros(7)
    twothreadtimeN=np.zeros(7)
    timef=np.zeros(7)
    timep=np.zeros(7)

    #choose M/N values
    Msize=[31,81,101,151,201,271,351]
    Nsize=[31,81,101,151,201,271,351]


    #caluclate runtimes for one thread (Fortran/OpenMP)
    tr.numthreads=1
    for i in range(7):
        t1=time.time()
        tr.simulate2_omp(51,50,Msize[i])
        t2=time.time()
        onethreadtimeM[i]=t2-t1
        t1=time.time()
        tr.simulate2_omp(Nsize[i],50,50)
        t2=time.time()
        onethreadtimeN[i]=t2-t1

    #calculate runtimes for two threads (Fortran/OpenMP) & Fortran, Python
    tr.numthreads=2
    for i in range(7):
        t1=time.time()
        tr.simulate2_omp(51,50,Msize[i])
        t2=time.time()
        twothreadtimeM[i]=t2-t1
        t1=time.time()
        tr.simulate2_omp(Nsize[i],50,50)
        t2=time.time()
        twothreadtimeN[i]=t2-t1
        t1=time.time()
        tr.simulate2_f90(51,50,Msize[i])
        t2=time.time()
        timef[i]=t2-t1
        t1=time.time()
        simulate2(51,50,1.1,0.01,0.95,Msize[i])
        t2=time.time()
        timep[i]=t2-t1

    #calculate speedup of Fortran/OpenMP
    speedupM=np.divide(onethreadtimeM,twothreadtimeM)

    #calculate relative performance of F/OpenMP and Fortran, Python
    relfortran=np.divide(twothreadtimeM,timef)
    relpython=np.divide(twothreadtimeM,timep)

    #plot figures
    if display == True:
        plt.figure()
        plt.plot(Msize, speedupM, 'r')
        plt.xlabel('M')
        plt.ylabel('Speedup')
        plt.title('Speedup of simulate2_omp against M')
        plt.show()

        plt.figure()
        plt.plot(Msize, onethreadtimeM, 'b', label='1 thread (M)')
        plt.plot(Nsize, onethreadtimeN, 'b--', label='1 thread (N)')
        plt.plot(Msize, twothreadtimeM, 'g', label='2 threads (M)')
        plt.plot(Nsize, twothreadtimeN, 'g--', label='2 threads (N)')
        plt.xlabel('M/N')
        plt.ylabel('Runtime')
        plt.legend()
        plt.title('Performance of simulate2_omp using 1 or 2 threads')
        plt.show()

        plt.figure()
        plt.plot(Msize, relfortran, 'r', label='Fortran')
        plt.plot(Msize, relpython, 'r--', label='Python')
        plt.xlabel('M')
        plt.ylabel('Runtime of simulate2_omp/Runtime of other function')
        plt.title('Relative Performance of Fortran/Python Implementations')
        plt.legend()
        plt.show()

    return None #Modify as needed

def analyze(input=(None),display=False):
    """Analyze influence of model parameter, g.
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed

    Comments:

    The first graph shows that the new variable, gamma, affects both
    the value that the fc_ave tends to, as well as the number of days
    the fc_ave takes to converge. Taking b=1.3, we can see that as gamma
    increases from 0.8, the proportions of M/C villages takes more time
    to become stable, but tend to result in M winning more of the villages
    (shown by the lower fc_ave). This is true except for gamma = 1, when
    the fc_ave appears to be tending to around a half.

    The second graph is a contour plot of fc_ave for different values of
    b and g. The yellow vertical line at gamma =1 corresponds to what we
    found in the previous graph, and we see that this high proportion of
    C villages occurs for all values of b. Except for this, we can see that
    for all values of b, M fares better as gamma increases, and has the best
    chance for high values of b and g (except 1).
    """

    #set parameters
    tr.tr_e = 0.01
    N=51
    tr.tr_b=1.3

    #choose gamma/b values
    gammas = [0.8, 0.9, 0.95, 0.98, 0.995, 0.9995, 1]
    bs = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5]

    #intialise array
    fcave80 = np.zeros((len(bs), len(gammas)))

    #plot figures
    if display == True:
        plt.figure()
        for i in range(len(gammas)):
            tr.tr_g=gammas[i]
            fcave = tr.simulate2_omp(N,100,100)[1]
            plt.plot(range(101),fcave, label='g ='+str(gammas[i]))
        plt.xlabel('Time')
        plt.ylabel('fc_ave')
        plt.title('fc_ave over time for different g values (b=1.3)')
        plt.legend()
        plt.show()

        plt.figure()
        X, Y = np.meshgrid(gammas, bs)
        for i in range(len(gammas)):
            tr.tr_g = gammas[i]
            for j in range(len(bs)):
                tr.tr_b=bs[j]
                fcave80[j,i]=tr.simulate2_omp(N,80,100)[1][40]
        plt.contourf(X,Y, fcave80, 15)
        plt.colorbar()
        plt.xlabel('g')
        plt.ylabel('b')
        plt.title('Contour plot of fc_ave after 80 days (e=0.01)')
        plt.show()

    return None #Modify as needed



def visualize():
    """Generate an animation illustrating the evolution of
        villages during C vs M competition
    """

    #set parameters
    tr.tr_e=0.01
    tr.tr_g=0.95
    tr.tr_b=1.2

    #get 3D array of one trial at each time step
    evolv= tr.simulate22_f90(21,100)

    def update_fig(x):
        im.set_array(evolv[:,:,x])
        return im,

    #get animation
    fig=plt.figure()
    im=plt.imshow(evolv[:,:,0], animated=True)
    x =0
    ani = animation.FuncAnimation(fig, update_fig, frames=100, blit=True, repeat=False)
    ani.save('hw3movie.mp4', writer='ffmpeg')   #save animation

    return None #Modify as needed


if __name__ == '__main__':
    #Modify the code here so that it calls performance analyze and
    # generates the figures that you are submitting with your code

    input_p = None
    output_p = performance(input_p) #modify as needed

    input_a = None
    output_a = analyze(input_a)
