"""M3C 2018 Homework 1 Frederica Melbourne 01068192
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def simulate1(N,Nt,b,e):
    """Simulate C vs. M competition on N x N grid over
    Nt generations. b and e are model parameters
    to be used in fitness calculations
    Output: S: Status of each gridpoint at tend of somulation, 0=M, 1=C
    fc: fraction of villages which are C at all Nt+1 times
    Do not modify input or return statement without instructor's permission.
    """
    #Set initial condition
    S = np.ones((N,N),dtype=int) #Status of each gridpoint: 0=M, 1=C
    j = int((N-1)/2)
    S[j-1:j+2,j-1:j+2] = 0
    
    fc = np.zeros(Nt+1)
    fc[0] = S.sum()/(N*N)
    
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) #kernel for adding up neighbours
    neighs = convolve(np.ones((N,N)), kernel, mode='constant')  #count of no. neighbours of each entry

    for k in range(Nt):
        C = (S==1)*convolve(S, kernel, mode='constant')    #get fitness for Cs
        switch = (S==0)*e + (S==1)*b
        M = (S==0)*convolve(switch, kernel, mode='constant') #get fitness for Ms
        fitness = (C+M)/neighs #matrix of all fitness values
        totalfit= convolve(fitness, np.ones((3,3)), mode='constant') 
        fitC = convolve(fitness*(S==1), np.ones((3,3)), mode='constant')
        P = fitC/totalfit  #matrix of probs of village becoming C
        S= np.random.binomial(1, P)
        fc[k+1] = S.sum()/(N*N) 

    return S,fc

def plot_S(S):
    """Simple function to create plot from input S matrix
    """
    ind_s0 = np.where(S==0) #C locations
    ind_s1 = np.where(S==1) #M locations
    plt.plot(ind_s0[1],ind_s0[0],'rs')
    plt.hold(True)
    plt.plot(ind_s1[1],ind_s1[0],'bs')
    plt.hold(False)
    plt.show()
    plt.pause(0.05)
    return None


def simulate2(N,Nt,b,e):
    """Simulation code for Part 2, add input variables as needed
    """
    #Set initial condition
    original = np.ones((N,N),dtype=int) #Status of each gridpoint: 0=M, 1=C
    j = int((N-1)/2)
    original[j-1:j+2,j-1:j+2] = 0
    fc = np.zeros((Nt+1,len(b)))
    
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) #kernel for adding up neighs
    neighs = convolve(np.ones((N,N)), kernel, mode='constant')  #count of no. neighs of each entry
    
    pctm = np.zeros(len(b))  #init list of fraction of times M wins
    pctc = np.zeros(len(b))  #init list of fraction of times C wins

    for i in range(len(b)):  #iterate over b values
        fc[0,i] = original.sum()*50/(N*N)
        for j in range(50):   #average over 50 simulations to remove noise    
            S=original  #start again from IC 
            for k in range(Nt):
                C = (S==1)*convolve(S, kernel, mode='constant')    #get fitness score for Cs
                switch = (S==0)*e + (S==1)*b[i]
                M = (S==0)*convolve(switch, kernel, mode='constant') #get fitness score for Ms
                fitness = (C+M)/neighs #all fitness scores
                totalfit= convolve(fitness, np.ones((3,3)), mode='constant') 
                fitC = convolve(fitness*(S==1), np.ones((3,3)), mode='constant')
                P = fitC/totalfit  #matrix of probs of village becoming C
                S= np.random.binomial(1, P)   
                fc[k+1,i] += S.sum()/(N*N)
                if k == Nt-1:
                    if fc[Nt,i] == 0:
                        pctm[i]+=1     
                    elif fc[Nt,i] == 1:
                        pctc[i] +=1
  
    fc=fc/50.0  
    pctc = pctc/50.0
    pctm = pctm/50.0
    pct = pctm + pctc #list of fraction of times the system becomes stable
        
    return fc, pctm, pctc, pct 


def analyze(b,Nt,line=False,bar=False,scatter=False):
    """ The first graph shows that for all values of b between 1.1 and 1.5, there is a tendency for M to win (occupy all villages). The fact that the initial gradient of the graph is more negative (steeper) the higher the value of b, shows that the greater b is, the faster M dominates. For the highest values of b, we can see that the gradient remains steep until M has nearly won. However, for the lower values, the time at which M begins to slow down (represented by the point of inflection) happens when the fraction of villages which belong to C is higher. 

The second graph shows that for values of b below 1.2, either of the sides winning within 100 years is very unlikely. As b increases past 1.2, the likelihood of the system reaching stability in this timeframe increases and for values of b over 1.45 almost all samples become stable. For values of b between 1.2 and 1.4, it appears that there is a more even likelihood of M or C winning.

The third and fourth graphs show that the proportion of M and C villages after 100 years for different b values follows a clear curve, meaning that this could be predicted accurately by fitting a curve. Increasing the no. samples from 50 to 100 causes a reduced variance of the points about this curve."""
    
    #fix parameters
    e=0.01
    N=21

    fracs, pctm, pctc, pct = simulate2(N, Nt, b, e)    
   
    if line == True:
        plt.xlabel('Nt')
        plt.ylabel('Average fc (50 samples)')
        plt.title('Freddie Melbourne, Analyze Function')
        for i in range(len(b)):
           s= str(b[i])
           plt.plot(fracs[:,i], label='b='+s)
        plt.legend()
        plt.show()
    
    if bar == True:
        fig, ax= plt.subplots()
        p1 = ax.bar(b, pctm, 0.015, color='r', bottom=0, label='M wins') #bar for fraction of times M wins
        p2 = ax.bar(b-0.015*np.ones(len(b)), pctc, 0.015, color='b', bottom=0, label='C wins') #bar for fraction of times C wins
        p3 = ax.bar(b+0.015*np.ones(len(b)), pct, 0.015, color='g', bottom=0, label='Either wins') #bar for total
        ax.set_title('Freddie Melbourne, Analyze Function')
        ax.set_xlabel('b')
        ax.set_ylabel('Fraction of 50 samples which are stable at Nt=100')
        ax.legend()
        plt.show()
    
    if scatter == True:
        plt.figure()
        plt.scatter(b, fracs[Nt,:], s=1.2)
        plt.xlabel('b')
        plt.ylabel('Average fc at Nt=100 (50 samples)')
        plt.title('Freddie Melbourne, Analyze Function')
        plt.show() 

if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    b1=np.linspace(1, 1.5, 6)
    b2=np.linspace(1, 1.6, 13)
    b3=np.linspace(1, 1.5, 126)
    Nt=100
    output = analyze(b1, Nt, line=True), analyze(b2, Nt, bar=True), analyze(b3, Nt, scatter=True) 
