# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:42:26 2023

@author: Augutin Guichardon & Valentin Sousa Bouquety
"""

import numpy as np
from math import*
import matplotlib.pyplot as plt

"""
Fonctions
"""
#Formule du rectangle à droite
def method_RD(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a+h,b,n)
    S = sum(f(x))
    return h*S

#Formule des trapèzes
def method_T(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a+h, b-h, n-1)
    S = sum(f(x)) + (f(a) + f(b))/2
    return h*S

#Runge-Kutta second order for iniline function & vector function
def ode_RK2(f, a, b, ic, N):
    h = (b - a) / N #step size if h is constant
    Lt = np.linspace(a, b, N)
    Ly = np.empty((N, np.size(ic)),dtype = float)
    Ly[0,:] = ic
    for i in range(N-1):
    #if h isn't constant, we use h = t[i+1]-t[i]
        k1 = h*f(Lt[i], Ly[i,:])
        k2 = h*f(Lt[i] + h/2, Ly[i,:]+k1/2)
        Ly[i+1,:] = Ly[i,:] + k2
    return (Lt, Ly)

#Euler method for vector functions F(t,y1,y2,.....)
#Ok for inline function
def ode_VectEulerExp(f, a, b, ic, N):
    h = (b - a) / N #step size if h is constant
    Lt = np.linspace(a, b, N)
    Ly = np.empty((N, np.size(ic)), dtype = float)
    Ly[0,:] = ic
    for i in range(N-1):
    #if h isn't constant, we use h=t[i+1]-t[i]
        Ly[i+1,:] = Ly[i,:] + h*f(Lt[i],Ly[i,:])
    return (Lt, Ly)
#Runge-Kutta fourth order for iniline function & vector function
def ode_RK4(f, a, b,ic, N):
    h = (b - a) / N #step size if h is constant
    Lt = np.linspace(a, b, N)
    Ly = np.empty((N, np.size(ic)),dtype = float)
    Ly[0,:] = ic
    for i in range(N-1):
    #if h isn't constant, we use h=t[i+1]-t[i]
        k1 = h*f(Lt[i], Ly[i,:])
        y1 = Ly[i,:] + 1/2*k1
        k2 = h* f(Lt[i]+h/2, y1)
        y2 = Ly[i,:] + 1/2*k2
        k3 = h* f(Lt[i]+h/2,y2)
        y3 = Ly[i,:] + k3
        k4 = h* f(Lt[i]+h, y3)
        k = (k1+2*k2+2*k3+k4)/6
        Ly[i+1,:] = Ly[i,:] + k
    return (Lt, Ly)

def f(t):
    return t**(3/2)*np.exp(-t/2)

def F(x):
    return (1/(3*sqrt(2*pi))*method_RD(f, 0, x, n))

def F1(x):
    return (1/(3*sqrt(2*pi))*method_T(f, 0, x, n))

def f2(t,Y):
    [y,dy] = Y
    return np.array([dy, sin(y) + sin(t)])

"""
Exercice 1
"""

listx=np.arange(0,20.1,0.1)  
listfx=[]
listf1x=[]
#print(listx) 

for x in listx:
    n=10
    Fx=round(F(x),5)
    F1x=round(F1(x),5)
#    print(Fx, F1x)
    listfx.append(Fx)
    listf1x.append(F1x)
    
plt.plot(listx, listfx)
plt.plot(listx, listf1x)
plt.title('F(x) en fonction de x')
plt.legend(["Méthode RD","Méthode des Trapèzes"],loc="lower right")
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid()
plt.show()

"""
Exercice 2
"""
a1 = 0
b1 = 10
ic2 = np.array([0,-1])
N=100
plt.figure(N+1)

T,Y= ode_VectEulerExp(f2, a1, b1, ic2, N)
T1,Y1=ode_RK2(f2, a1, b1, ic2, N)
T2,Y2=ode_RK4(f2, a1, b1,ic2, N)
Y = Y[:,0]
Y1 = Y1[:,0]
Y2 = Y2[:,0]

plt.plot(T,Y,label = 'Explicit Euler with N='+ str(N))
plt.plot(T1,Y1,label = 'Runge-Kutta2 with N='+ str(N))
plt.plot(T2,Y2,label = 'Runge-Kutta4 with N='+ str(N))

plt.xlabel('t')
plt.ylabel('y and dy/dt')
plt.title("Comparison methods-solving ODE: y''=sin(y) + sin(t) on [0;10]")
plt.legend()
plt.grid()
plt.show()






   
