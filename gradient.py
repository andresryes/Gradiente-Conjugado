#Imports 
import numpy as np
from numpy import linalg
import scipy as sp
from scipy import optimize
import math
#Gradiente conjugado
#Esta es la función a optimizar
def f(x):
    return 0.5*x[0]**2 + 0.5*x[1]**2 - 5.3*x[0] - 6.2*x[1]
#El gradiente de la función, indica la dirección
def gradiente(x):
    y=[x[0]-5.3,0.6*x[1]-6.2]
    return y  
#Función alpha de una variable, indica la magnitud de los saltos
def alpha(alpha,x,s):
    y = 0.5*(x[0]+alpha*s[0])**2 + 0.5*(x[1]+alpha*s[1])**2 - 5.3*(x[0]+alpha*s[0]) - 6.2*(x[1]+alpha*s[1])
    return y
#El método en sí, siguiendo el algoritmo
#Este es el metodo del descenso mas pronunciado    
def conjugado(xk,f,tol):
    #Error igual a algo mayor para que cumpla la condición
    #inicial
    e=1
    #Arreglo con todos los puntos
    X=[]
    #Arreglo con las direcciones calculadas
    S=[]
    #XK recibe el primer valor de x
    X.append(xk)
    #el gradiente que servirá para calcular la nueva dirección
    gk = gradiente(xk)
    #Nuestra primera dirección, es el negativo del gradiente
    sk=np.dot(-1.0,gradiente(xk))  
    while(e>tol):
        #se optimiza alpha, utilizando Golden Ratio de python
        ak1=sp.optimize.golden(alpha,args=(xk,sk))
        #Obtengo mi nuevo paso, al multiplicar la dirección 
        # y la magnitud alpha
        sk1=np.dot(ak1,sk)
        #Guarda la dirección en el arreglo
        S.append(sk1)
        #Se calcula la nueva x
        xk1=xk+np.dot(ak1,sk1)
        #Se guarda en el arreglo de X
        X.append(xk1)
        #Se calcula el error 
        e=np.linalg.norm(xk1-xk)
        #Mi nueva x es ahora la x del algoritmo
        xk=xk1
        #mi nuevo gradiente al evaluar la función en la nueva x
        gk1 = gradiente(xk)
        #la magnitud de cambio bk es calculada utilizando g0 y g1
        bk1 = np.dot(np.transpose(gk1),gk1)/np.dot(np.transpose(gk), gk)
        #obtengo una nueva dirección
        sk1 = np.dot(-1.0,gk1) + np.dot(bk1,sk)
        #mi dirección anterior se convierte en la nueva para los cálculos
        #del algoritmo
        sk = sk1
    #esta función regresa el vector de coordenadas, los arreglos 
    #de dirección y de 
    #coordenadas anteriores    
    return xk,X,S
y=conjugado([7,0],f,0.000001)[0] 
print("La coordenada óptima es: "+str(y))