import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#a = 0.05 # k_tc
#b = 0.05 # k_ct



# definition of 

#pct = 0.0012
#ptc = 15.6666666667  * pct

pct =  0.0012 #/1000
ptc =  0.0188 #/1000


#ptc =  0.01 #0.188/1000
#pct =  0.3

# steady-state solution (binomial distribution)

ttt, ttc, tcc, ccc = 1 * pct**3 / (pct+ptc)**3 , 3* pct**2 * ptc / (pct+ptc)**3, 3* pct * ptc**2 / (pct+ptc)**3, 1*ptc**3 / (pct+ptc)**3

print(ttt,ttc,tcc,ccc)



# numerical solution of linear ODE system


#a = ptc/108*2 #1.88/108*2 # k_tc
#b = pct/108*2 #a / (5.3) #0.12/108*2 # k_ct

a = ptc #/100
b = pct #/100

#N = 1.08 # N azo

a1 = ptc*3 
a2 = ptc*2 
a3 = ptc

b1 = pct
b2 = pct*2 
b3 = pct*3 




#def dydt(P, t):
#    coeff_matrix = ()
#    return 

def dx_dt(x, t):
    #return [-a*x[0] + b*x[1],   a*x[0] - (a+b)*x[1] + b * x[2], a*x[1] - (a+b)*x[2] + b * x[3], a *x[2] - b * x[3]]
    #return [-a*x[0] + b*x[1],  a*x[0] - (a+b)*x[1] + b * x[2], a*x[1] - (a+b)*x[2] + b * x[3], a *x[2] - b * x[3]]
    #return [-3*a*x[0] + b*x[1],  2*a*x[0] - (a+b)*x[1] + b * x[2], a*x[1] - (a+b)*x[2] + b * x[3], a *x[2] - b * x[3]]
    #return [-a1*x[0] + b1*x[1],  a1*x[0] - (a2+b1)*x[1] + b2 * x[2], a2*x[1] - (a3+b2)*x[2] + b3 * x[3], a3 *x[2] - b3 * x[3]]
    return [-3*a*x[0] + b*x[1],  3*a*x[0] - (2*a+b)*x[1] + 2*b * x[2], 2*a*x[1] - (a+2*b)*x[2] + 3*b*x[3], a*x[2] - 3*b*x[3]]


ts = np.linspace(0, 500, 5000)

x0 = [1.0, 0.0, 0.0, 0.0]
#x0 = [0.930, 0.07, 0.0, 0.0]
#x0 = [0.00, 0.01, 0.16, 0.83]

xs = odeint(dx_dt, x0, ts)
x_ttt = xs[:,0]
x_ttc = xs[:,1]
x_tcc = xs[:,2]
x_ccc = xs[:,3]





#predators = Ps[:,1]


#prey = Ps[:,0]
#predators = Ps[:,1]


#Ps = odeint(dP_dt, P0, ts)
#prey = Ps[:,0]
#predators = Ps[:,1]

#a,b,c,d = 1,1,1,1

#def dP_dt(P, t):
#    return [P[0]*(a - b*P[1]), -P[1]*(c - d*P[0])]

#ts = np.linspace(0, 12, 100)
#P0 = [1.5, 1.0]
#Ps = odeint(dP_dt, P0, ts)
#prey = Ps[:,0]
#predators = Ps[:,1]

print(x_ttt[-1], x_ttc[-1], x_tcc[-1], x_ccc[-1])
print(x_ttt[-1]+ x_ttc[-1]+x_tcc[-1]+x_ccc[-1])

plt.plot(ts, x_ttt,  label="ttt", c = "xkcd:electric blue")
plt.plot(ts, x_ttc, label="ttc", c="xkcd:turquoise")
plt.plot(ts, x_tcc,  label="tcc", c = "xkcd:orange yellow")
plt.plot(ts, x_ccc,  label="ccc", c = "xkcd:orange red")
plt.xlabel("Time (or Cycles)")
plt.ylabel("Numer Fractions")
plt.legend();
#plt.show()




# EXACT SOLUTION


def frac_ttt(t, a, b):
    return a**3 / (a+b)**3 * ((b**3)/(a**3) + 3*(b**2)/(a**2) * np.exp(-1.*(a+b)*t) + 3*b/a * np.exp(-2.*(a+b)*t)  + np.exp(-3.*(a+b)*t))

def frac_ttc(t, a, b):
    return a**3 / (a+b)**3 * ( 3*(b**2)/(a**2) + 3*(2*a*b - b**2)/(a**2) * np.exp(-1.*(a+b)*t) + 3*(a-2*b)/a* np.exp(-2.*(a+b)*t)  - 3 * np.exp(-3.*(a+b)*t))


def frac_tcc(t, a, b):
    return a**3 / (a+b)**3 * (3*b/a + 3*(a-2*b)/a * np.exp(-1.*(a+b)*t) + 3*(b-2*a)/a * np.exp(-2.*(a+b)*t)  + 3 * np.exp(-3.*(a+b)*t))


def frac_ccc(t, a, b):
    return a**3 / (a+b)**3 * (1  - 3 * np.exp(-1.*(a+b)*t) + 3 * np.exp(-2.*(a+b)*t) - 1 * np.exp(-3.*(a+b)*t))



ttt_vec = np.zeros(500)
ttc_vec = np.zeros(500)
tcc_vec = np.zeros(500)
ccc_vec = np.zeros(500)
print(ttt_vec)




for t in range(0,500):
    ttt_vec[t] = frac_ttt(t, a, b)
    ttc_vec[t] = frac_ttc(t, a, b)
    tcc_vec[t] = frac_tcc(t, a, b)
    ccc_vec[t] = frac_ccc(t, a, b)


T = np.arange(0,500)


plt.plot(T, ttt_vec,  label="ttt (exact)",  marker="+", c = "xkcd:electric blue")
plt.plot(T, ttc_vec,  label="ttc (exact)",  marker="+", c = "xkcd:turquoise")
plt.plot(T, tcc_vec,  label="tcc (exact)",  marker="+", c = "xkcd:orange yellow")
plt.plot(T, ccc_vec,  label="ccc (exact)",  marker="+", c = "xkcd:orange red")

plt.show()

exit()

##### exact solution

def x_s(C, A, t):
    c0, c1, c2, c3 = C
    print("coeffs:")
    print(c0, c1, c2, c3)
    print(A[:,0])
    evec0 = A[:,0] #np.array([])
    evec1 = A[:,1]
    evec2 = A[:,2]
    evec3 = A[:,3]
    
    eval0 = 0
    eval1 = -a-b
    eval2 = -3*(a+b)
    eval3 = -2*(a+b)

    sol = c0 * np.exp(eval0*t) * evec0 + c1 * np.exp(eval1*t) * evec1 + c2 * np.exp(eval2*t) * evec2 + c3 * np.exp(eval3*t) * evec3
    #print("solution")
    #print(sol)
    return sol


#a = np.array([[1, 2], [3, 5]])
A = np.array([[b**3/a**3, -b**2/a**2, -1, b/a                 ],
              [3*b**2/a**2, -(2*a*b-b**2)/a**2, 3, -(-a+2*b)/b],
              [3*b/a, -(a-2*b)/a, -3, -(2*a-b)/a                  ],
              [1,1,1,1                                        ]])

#A = np.zeros((4,4))
#
#A[0,0] = b**3/a**3
#A[0,1] = -b**2/a**2
#A[0,2] = -1
#A[0,3] = b/a
#
#A[1,0] = 3*b**2/a**2
#A[1,1] = -(2*a*b-b**2)/a**2
#A[1,2] =  3
#A[1,3] = -(-a+2*b)/b
#
#
#A[1,0] = 3*b/a
#A[1,1] = -(a-2*b)/a
#A[1,2] = -(2*a-b)/a
#A[1,3] = 


#A[1,0] = 
#A[1,1] = 
#A[1,2] = 
#A[1,3] = 



#A = np.array([[b**3/a**3, -b**2/a**2, -1, b/a],
#[3*b**2/a**2, -(2*a*b-b**2)/a**2, 3, -(-a+2*b)/b],
#[3*b/a, -(a-2*b)/a, -(2*a-b)/a],
#[1,1,1,1]])

b = np.array([1, 0, 0, 0])
#A = A.reshape(4,4)
print(A)
print(A.shape)
print(A.size)

print(b)
C = np.linalg.solve(A, b)


print(C)
x_vector = np.zeros((4, 500))
for t in range(0,500):
    x_vector[:,t] = x_s(C, A, t)
   
T = np.arange(0,500)

#print(x_vector)
#plt.plot(T[:], x_vector[0,:])
#plt.plot(T[:], x_vector[1,:])
#plt.plot(T[:], x_vector[2,:])
#plt.plot(T[:], x_vector[3,:])

#plt.show()



