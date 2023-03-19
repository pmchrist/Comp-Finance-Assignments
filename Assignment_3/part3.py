import numpy as np
import matplotlib.pyplot as plt


params = {'r':[0.04,0.04,0.04],
          'sigma':[0.3,0.3,0.3],
          'S_0':[100,110,120],
            'K':[110,110,110],
            'T':[1,1,1]}

def fourier_coefficients(a,b,psi_k,X_k, K):

    return 2/(b-a)*K*(X_k-psi_k)


def x_k(a,b,c,d,k):
    component_1 = np.cos(k*np.pi*(d-a)/(b-a))*np.e**d - np.cos(k*np.pi*(c-a)/(b-a))*np.e**c
    component_2 = (np.sin(k*np.pi*(d-a)/(b-a))*np.e**d - np.sin(k*np.pi*(c-a)/(b-a))*np.e**c)*k*np.pi/(b-a)
    return (component_1 + component_2)/(1+(k*np.pi/(b-a))**2)

def psi_k(a,b,c,d,k):
    arr_return = np.zeros(len(k))
    arr_return[0] = d-c

    arr_return[1:] = (np.sin(k[1:]*np.pi*(d-a)/(b-a))- np.sin(k[1:]*np.pi*(c-a)/(b-a))*np.e**c)*(b-a)/(k[1:]*np.pi)

    return arr_return
def a(S_0,K,T,r,sigma):
    return np.log(S_0/K)+r*T - 12*np.sqrt(sigma**2*T)

def b(S_0,K,T,r,sigma):
    return np.log(S_0/K)+r*T + 12*np.sqrt(sigma**2*T)
def characteristic_func(u,sigma,r,t):
    return np.e**(1j*u*(r-0.5*sigma**2)*t - 0.5*sigma**2*t*u**2)

# Note: x must be S_0/K
def F_n(a,b,n,sigma,r,t):

    real_portion = np.real(characteristic_func(n*np.pi/(b-a),sigma,r,t)*np.e**(-1j*n*np.pi*a/(b-a)))

    return 2/(b-a)*real_portion



if __name__ == '__main__':
    N_vals = [64,96,128,160,192]
    T=1
    # k=np.linspace(0,64,65)
    time_steps = np.linspace(0,1,360)
    num_experiments = 3
    for i in range(num_experiments):
        for j in N_vals:
            k = np.linspace(0, j, j+1)
            all_t_vals = []
            for t in time_steps:
                a_ = a(params['S_0'][i],params['K'][i],params['T'][i],params['r'][i],params['sigma'][i])
                b_ = b(params['S_0'][i],params['K'][i],params['T'][i],params['r'][i],params['sigma'][i])
                comp_1 = x_k(a_,b_,0,b_,k)
                comp_2 = psi_k(a_,b_,0,b_,k)
                G_k = fourier_coefficients(a_,b_,comp_1,comp_2,params['K'][i])

                approx_S_t = np.sum(F_n(a_, b_, k, params['sigma'][i], params['r'][i], T-t)*G_k)*np.e**(-params['r'][i]*(T-t))
                all_t_vals.append(approx_S_t)
            plt.plot(time_steps,all_t_vals,label=f'N={j}')

        plt.legend()
        plt.show()