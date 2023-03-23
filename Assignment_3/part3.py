import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
import scipy as sp
import random
def fourier_coefficients_eu_call(a,b,psi_k,X_k, K):

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
    real_portion = characteristic_func(u,sigma,r,t)

    return real_portion

def d1_d2(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1,d2

def call_option_price(S0, K, r, T, d1, d2):
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_delta(d1):
    return norm.cdf(d1)




def run_part_3(N_vals,params):
    random.seed(42)
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

            n_vals.append(np.log(abs(bs_price-approx_S_t)))
            times.append(np.log(end-start))
        all_times.append(times)
        print(n_vals)
        print(f'Times: {times}')

        plt.plot(N_vals, n_vals, label=f'Initial Price: {params["S_0"][i]}')
    plt.legend()
    plt.xlabel('Number of Coefficients')
    plt.ylabel('log(Error)')
    plt.title('log(Error) vs Number of Coefficients')
    plt.grid()
    plt.show()
    for i in range(len(all_times)):
        plt.plot(N_vals, all_times[i], label=f'Initial Price: {params["S_0"][i]}')

    plt.xlabel('Number of Coefficients')
    plt.ylabel('log(Time) (log(s))')
    plt.title('Time vs Number of Coefficients')
    plt.legend()
    plt.grid()
    plt.show()


    # plt.plot(N_vals,all_experiments,label=f'N_vals for experiment {i+1}')
    # # plt.plot(N_vals,bs_price*np.ones(len(N_vals)),label='Black Scholes')
    # plt.title(f'Experiment {i+1}')
    # plt.legend()
    # plt.show()



def FTCS_delta(a_mat,S_M,K,r,T,sigma, M,N, alpha=10):
    x_M = np.log(S_M)
    dx = S_M/M
    dt = T/N
    ds = np.e**dx
    k_1 = np.zeros(M)
    k_1[-1] = S_M*((r-0.5*sigma**2)+sigma**2/dx)

    # dt = dx/alpha
    v_arr = np.zeros((N, M))
    init_price = np.maximum(np.linspace(0, S_M, M) - K, 0)
    v_arr[0, :] = init_price # initial condition

    v_arr[:, -1] = S_M-K*np.e**(-r*T) #Boundary condition
    delta_arr = np.zeros((N, M))
    # Create the grid

    for n in range(0,N-1):
        for i in range(1,M-1):
            comp_1 =v_arr[n][i]
            comp_2 = (r - 0.5 * sigma ** 2) * dt / 2 / dx * (v_arr[n][i + 1] - v_arr[n][i - 1])
            comp_3 = 0.5 * sigma ** 2 * dt / dx ** 2 * (v_arr[n][i + 1] - 2 * v_arr[n][i] + v_arr[n][i - 1])
            comp_4 = -r*dt*v_arr[n][i]
            v_arr[n+1][i] = comp_1 + comp_2 + comp_3 + comp_4
            # delta_arr[0][i] = (0.5/dx)*(np.dot(a_mat[:,i],v_arr[:,i]))+k_1[i]/i
    prior = v_arr[:, -2]
    v_arr[:, -1] = prior
    delta_arr = delta_calc(v_arr, dx)
    return v_arr,delta_arr

def crank_nicolson(a_mat,S_M,K,r,T,sigma, M,N, alpha=10):
    x_M = np.log(S_M)
    dx = S_M/M
    # dx = x_M/M
    dt = T/N
    # dt = T/N
    ds = np.e**dx
    k_1 = np.zeros(M)
    k_1[-1] = S_M*((r-0.5*sigma**2)+sigma**2/dx)


    v_arr = np.zeros((N, M))
    init_price = np.maximum(np.linspace(0, M, M) - K, 0)
    v_arr[0, :] = init_price # initial condition
    # v_arr[:, -1] = S_M-K*np.e**(-r*T) #Boundary condition

    v_arr[:, -1] = S_M-K*np.e**(-r*T) #Boundary condition
    delta_arr = np.zeros(M)

    for n in range(0, N - 1):
        # v_arr[n, -1] = S_M - K * np.e ** (-r * (T-n*dt))  # Boundary condition
        for i in range(1, M - 1):
            comp_1 = v_arr[n][i]
            # comp_2 = 0.25*sigma**2*dt/dx*(v_arr[n][i+1]-2*v_arr[n][i]+v_arr[n][i-1] + v_arr[n+1][i+1]-2*v_arr[n+1][i] + v_arr[n+1][i-1])
            # comp_3 = 0.25*r*dt/dx*(v_arr[n][i+1]-v_arr[n][i-1] + v_arr[n+1][i+1]-v_arr[n+1][i-1])
            comp_2 = (r-0.5*sigma**2)*dt/4/dx*(v_arr[n][i+1] - v_arr[n][i-1] + v_arr[n+1][i+1] - v_arr[n+1][i-1])
            comp_3 = 0.25*sigma**2*dt/dx**2*(v_arr[n][i+1] - 2*v_arr[n][i] + v_arr[n][i-1] + v_arr[n+1][i+1] - 2*v_arr[n+1][i] + v_arr[n+1][i-1])
            comp_4 = -0.5*r*dt*(v_arr[n][i]+v_arr[n+1][i])


            v_arr[n+1][i] = comp_1 + comp_2 + comp_3 + comp_4

            # delta_arr[i] = (0.5 / dx) * (np.dot(a_mat[:, i], v_arr[:, i])) + k_1[i] / i

    prior = v_arr[:, -2]
    v_arr[:, -1] = prior

    delta_arr = delta_calc(v_arr,x_M/M)

    return v_arr,delta_arr

def finite_diff_mats(N,sigma,r,dt, d_X):

    A_1 = sp.sparse.diags([-1,0, 1], [-1, 0, 1], shape=(N,N)).toarray()
    A_2 = sp.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N,N)).toarray()

    middle = 1-dt*sigma**2/(d_X**2)-r*dt
    side = 0.5*dt*(sigma**2/d_X**2+(r-0.5*sigma**2)/d_X)

    A_3 = sp.sparse.diags([side,middle,side],[-1, 0, 1],shape=(N,N)).toarray()
    A_3[0, 0] = 1 - r * dt
    A_3[0, 1] = 0
    A_3[1, 0] = (dt * sigma ** 2) / d_X ** 2
    A_3[-1, -1] = 0
    return (A_1, A_2, A_3)


def delta_calc(v_arr, dx):
    delta_arr = []
    for j in range(1, v_arr.shape[0] - 1):
        delta_arr.append((v_arr[j + 1, :] - v_arr[j - 1,:]) / dx / 2)
    return delta_arr

if __name__ == '__main__':
    params = {'r': [0.04, 0.04, 0.04],
              'sigma': [0.3, 0.3, 0.3],
              'S_0': [100, 110, 120],
              'K': [110, 110, 110],
              'T': [1, 1, 1]}

    N_vals = [8,16,32,64,128,160,192]
    run_part_3(N_vals, params)



    # N=200
    # M=200
    # vol=0.3
    # r=0.04
    # dt=0.01
    # d_X=0.01
    #
    # S_M = 200
    # K  = 110
    # num_exp = 3
    # option_prices = []
    # final_plts = []
    # crank = []
    # delta_crank = []
    # delta_ftcs = []
    # bs_deltas = []
    # for i in range(num_exp):
    #
    #     dt = params['T'][i]/N
    #     d_X = S_M/M
    #     fd_1, fd_2, fd_3 = finite_diff_mats(N, params['sigma'][i], params['r'][i], dt, d_X)
    #
    #     v_arr, delta = FTCS_delta(fd_1, S_M, params['K'][i], r, params['T'][i], params['sigma'][i], M, N, alpha=10)
    #     final_plts.append(v_arr[-1, :])
    #     delta_ftcs.append(delta[-1])
    #     # plt.plot(np.linspace(0, M, M), v_arr[-1, :][:M], label=f'Experiment {i+1}')
    #     crank_nic,delta_cn = crank_nicolson(fd_1, S_M, params['K'][i], r, params['T'][i], params['sigma'][i], M, N, alpha=10)
    #     crank.append(crank_nic[-1,:])
    #     delta_crank.append(delta_cn[-1])
    #
    #     #FTCS 3DGraph
    #     # fig = plt.figure(figsize=(16, 8))
    #     # ax = fig.add_subplot(projection='3d')
    #     #
    #     # # Make data.
    #     # time = np.arange(0, N, 1)
    #     # price = np.arange(0, M, 1)
    #     # time, price = np.meshgrid(time, price)
    #     #
    #     # # Plot the surface.
    #     # surf = ax.plot_surface(time,price, v_arr, cmap= 'plasma')
    #     #
    #     # ax.set_xlabel('Time Steps')
    #     # ax.set_ylabel('Asset Price')
    #     # ax.set_zlabel('Option Price')
    #     # ax.view_init(20, 260)
    # # plt.legend()
    # S_0 = np.linspace(0, M, M)
    # d1,d2 = d1_d2(S_0, params['K'][0], params['r'][0], params['sigma'][0], params['T'][0])
    #
    # bs_deltas.append(bs_delta(d1))
    # option_prices = call_option_price(S_0, K, r, params['T'][0], d1, d2)
    # plt.plot(np.linspace(0, M, M), final_plts[0], label=f'Experiment {1}')
    # plt.plot(S_0, option_prices, c='r')
    # plt.legend()
    # plt.title('FTCS')
    # plt.show()
    #
    # plt.plot(S_0, option_prices, c='r')
    # plt.plot(np.linspace(0, M, M), crank[0], label=f'Experiment {1}')
    # plt.title('Crank-Nicolson')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(S_0,delta_crank[0],label='Crank-Nicolson')
    # plt.plot(S_0,delta_ftcs[0],label='FTCS')
    # plt.plot(S_0,bs_deltas[0],label='Black-Scholes')
    # plt.legend()
    # plt.title('Delta')
    # plt.show()
