import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time

def fourier_coefficients_eu_call(a,b,psi_k,X_k, K):
    # test_val = np.cos(k*np.pi*(x-a)/(b-a))
    # return test_val
    return 2/(b-a)*(X_k-psi_k)*K


def x_k(a,b,c,d,k):
    component_1 = np.cos(k*np.pi*(d-a)/(b-a))*np.e**d - np.cos(k*np.pi*(c-a)/(b-a))*np.e**c
    component_2 = (np.sin(k*np.pi*(d-a)/(b-a))*np.e**d - np.sin(k*np.pi*(c-a)/(b-a))*np.e**c)*k*np.pi/(b-a)

    return (component_1 + component_2)/(1+(k*np.pi/(b-a))**2)

def psi_k(a,b,c,d,k):
    arr_return = np.zeros(len(k))
    arr_return[0] = d-c

    arr_return[1:] = (np.sin(k[1:]*np.pi*(d-a)/(b-a))- np.sin(k[1:]*np.pi*(c-a)/(b-a)))*(b-a)/(k[1:]*np.pi)

    return arr_return
def a(S_0,K,T,r,sigma):
    return np.log(S_0/K)+r*T - 12*np.sqrt(sigma**2*T)

def b(S_0,K,T,r,sigma):
    return np.log(S_0/K)+r*T + 12*np.sqrt(sigma**2*T)
def characteristic_func(u,sigma,r,t):
    return np.e**(1j*u*(r-0.5*sigma**2)*t - 0.5*sigma**2*t*u**2)

# Note: x must be S_0/K
def F_n(a,b,n,sigma,r,t,x):
    u = n*np.pi/(b-a)
    real_portion = characteristic_func(u,sigma,r,t)#*np.exp(1j * np.outer((x - a), n*np.pi/(b-a)))#*np.e**(-1j*n*np.pi*a/(b-a))

    return real_portion#*np.cos(n*np.pi*(x-a)/(b-a))

def d1_d2(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1,d2

def call_option_price(S0, K, r, T, d1, d2):
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)




if __name__ == '__main__':
    params = {'r': [0.04, 0.04, 0.04],
              'sigma': [0.3, 0.3, 0.3],
              'S_0': [100, 100, 100],
              'K': [110, 120, 130],
              'T': [1, 1, 1]}

    N_vals = [8,16,32,64,128,160,192]
    num_experiments = 3
    bs_vals = []
    all_times = []
    for i in range(num_experiments):
        # Black Scholes
        d1, d2 = d1_d2(params['S_0'][i], params['K'][i], params['r'][i], params['sigma'][i], params['T'][i])
        bs_price = call_option_price(params['S_0'][i], params['K'][i], params['r'][i], params['T'][i], d1, d2)

        bs_vals.append(bs_price)

        n_vals = []
        times = []
        for j in N_vals:
            start = time()
            k = np.linspace(0, j-1, j)
            a_val = a(params['S_0'][i], params['K'][i], params['T'][i], params['r'][i], params['sigma'][i])
            b_val = b(params['S_0'][i], params['K'][i], params['T'][i], params['r'][i], params['sigma'][i])
            comp_1 = x_k(a_val, b_val, 0, b_val, k)
            comp_2 = psi_k(a_val, b_val, 0, b_val, k)
            G_k = fourier_coefficients_eu_call(a_val, b_val, comp_2, comp_1, params['K'][i])

            F_k = F_n(a_val,b_val,k,params['sigma'][i],params['r'][i],params['T'][i], params['S_0'][i]/params['K'][i])

            mult_vals = G_k*F_k
            mult_vals[0] = mult_vals[0]*0.5

            all_exp = np.exp(1j * np.outer((np.log(params['S_0'][i]/params['K'][i]) - a_val), k * np.pi / (b_val - a_val)))

            approx_S_t = np.real(np.dot(all_exp, mult_vals)) * np.e**(-params['r'][i]*(params['T'][i]))
            end = time()
        # print(f'Approx S_t: {approx_S_t} \n'
        #       f'BS Price: {bs_price} \n'
        #       f'Error: {abs(approx_S_t-bs_price)} \n'
        #       f'N: {j}\n'
        #       f'a: {a_val}\n'
        #       f'b: {b_val}\n'
        #       f'G_k*F_k: {np.sum(G_k*F_k)}\n'
        #       )
        # all_t_vals.append(approx_S_t)
            n_vals.append(np.log(abs(bs_price-approx_S_t)))
            times.append(end-start)
        all_times.append(times)
        print(n_vals)
        print(f'Times: {times}')

        plt.plot(N_vals, n_vals, label=f'Experiment {i+1}')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.title('log(Error) vs N')
    plt.grid()
    plt.show()
    for i in range(len(all_times)):
        plt.plot(N_vals, all_times[i], label=f'Experiment {i+1}')

    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.title('Time vs N')
    plt.legend()
    plt.grid()
    plt.show()
        # plt.plot(N_vals,all_experiments,label=f'N_vals for experiment {i+1}')
        # # plt.plot(N_vals,bs_price*np.ones(len(N_vals)),label='Black Scholes')
        # plt.title(f'Experiment {i+1}')
        # plt.legend()
        # plt.show()

