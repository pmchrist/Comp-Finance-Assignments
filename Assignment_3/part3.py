import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
import scipy as sp
import random
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
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

def CN_scheme(option_price,M,K):
    idx = np.arange(1, M)

    d3 = dt/4*(r*idx-sigma**2*idx**2)
    d2 = dt/2*(r+sigma**2*idx**2)
    d1 = -dt/4*(r*idx+sigma**2*idx**2)

    A_star = sp.sparse.diags([-d1[:-1],1-d2,-d3[1:]], [1, 0, -1]).toarray()
    all_CN = []
    all_deltas = []
    for i in range(N):
        A_1 = np.dot(A_star, option_price[1:-1])
        # add BC for the right bound (the last element)
        A_1[-1] += -2 * d1[-1] * (M - K)

        A_2 = sp.sparse.diags([d1[:-1],1+d2,d3[1:]], [ 1,0,-1]).toarray()

        option_price[1:-1] = np.linalg.inv(A_2)@A_1
        copy_of_price = np.copy(option_price)
        all_CN.append(copy_of_price)
        delta = (copy_of_price[2:] - copy_of_price[:-2]) / 2
        all_deltas.append(delta)

    return all_CN,all_deltas

def FTCS_scheme(C, N, M, dt, r, sigma):

    idx = np.arange(1, M)
    all_FTCS = []
    all_ftcs_deltas = []
    for n in range(N):
        comp_1 = C[1:-1]
        comp_2 = (r - 0.5 * sigma ** 2) *  0.5*dt* idx * (C[2:] - C[0:-2])
        comp_3 = 0.5 * sigma ** 2 *dt* idx ** 2 * (C[2:] - 2 * C[1:-1] + C[0:-2])
        comp_4 = -r * C[1:-1]  *dt

        C[1:-1] = comp_1 +comp_2+comp_3+ comp_4
        duplicate_c = C.copy()
        all_FTCS.append(duplicate_c)
        delta = (duplicate_c[2:] - duplicate_c[:-2]) / 2
        all_ftcs_deltas.append(delta)

    return all_FTCS, all_ftcs_deltas




if __name__ == '__main__':
    params = {'r': [0.04, 0.04, 0.04],
              'sigma': [0.3, 0.3, 0.3],
              'S_0': [100, 110, 120],
              'K': [110, 110, 110],
              'T': [1, 1, 1]}

    N_vals = [8,16,32,64,96,128,160,192]
    # run_part_3(N_vals, params)

    N=20000
    M=200
    vol=0.3
    r=0.04
    dt=0.01
    d_X=0.01
    sigma = 0.3

    S_M = 200
    K  = 110
    num_exp = 3

    # for i in range(num_exp):
    T =1
# number of space grids
    dt = T / N  # time step
    # stock_price = np.linspace(0, M, M + 1)
    # time_arr = np.linspace(0, T, N )
    # call_price = np.clip(stock_price - K, 0, M - K)
    # call_price_1 = np.clip(stock_price - K, 0, M - K)
    # cn_times = []
    # ftcs_times = []
    # start_cn = time()
    # CN_option_prices,CN_deltas = CN_scheme(call_price,M, K)
    # end_cn = time()
    # cn_times.append(np.log((end_cn-start_cn)/(M+1)))
    # # print(f'CN time: {np.log((end_cn-start_cn)/(M+1))}')
    # start_ftcs = time()
    # FTCS_option_prices,FTCS_deltas = FTCS_scheme(call_price_1, N, M, dt, r, sigma)
    # end_ftcs = time()
    # # print(f'FTCS time: {np.log((end_ftcs-start_ftcs)/(M+1))}')
    # ftcs_times.append(np.log((end_ftcs-start_ftcs)/(M+1)))


    all_times = []

    n_vals = []




    # bs_vals.append(bs_price)
    average_cos_times = []
    stock_price_points = np.array([100,110,120])
    all_cos_results = []
    all_avg_times = []
    for price in stock_price_points:
        n_results = []
        times = []
        for j in N_vals:
            i = 0
            d1, d2 = d1_d2(price, params['K'][i], params['r'][i], params['sigma'][i], params['T'][i])
            bs_price = call_option_price(price, params['K'][i], params['r'][i], params['T'][i], d1, d2)
            start = time()
            k = np.linspace(0, j-1, j)
            a_val = a(price, params['K'][i], params['T'][i], params['r'][i], params['sigma'][i])
            b_val = b(price, params['K'][i], params['T'][i], params['r'][i], params['sigma'][i])
            comp_1 = x_k(a_val, b_val, 0, b_val, k)
            comp_2 = psi_k(a_val, b_val, 0, b_val, k)
            G_k = fourier_coefficients_eu_call(a_val, b_val, comp_2, comp_1, params['K'][i])

            F_k = F_n(a_val, b_val, k, params['sigma'][i], params['r'][i], params['T'][i],
                      price / params['K'][i])

            mult_vals = G_k * F_k
            mult_vals[0] = mult_vals[0] * 0.5

            all_exp = np.exp(1j*(np.log(price/params['K'][i])-a_val)*k*np.pi/(b_val-a_val))

            approx_S_t = np.real(np.dot(all_exp, mult_vals))*np.e**(-params['r'][i]*(params['T'][i]))
            n_results.append(approx_S_t)
            end = time()
            n_vals.append(np.log(abs(bs_price - approx_S_t)))
            times.append(np.log(end - start))
        all_cos_results.append(n_results)

        all_times.append(times)

        #average time for each N
        # for i in range(len(all_times[0])):
        #     all_avg_times.append(np.mean([all_times[0][i],all_times[1][i],all_times[2][i]]))
            # print(f'Average time for N value {N_vals[i]} is {np.mean([all_times[0][i],all_times[1][i],all_times[2][i]])}')
        # average_cos_times.append(all_avg_times)

    # for i in range(len(average_cos_times[0])):
    #     print(f'Average time for N value {N_vals[i]} is {np.mean(average_cos_times[:][i])}')
    #     print(f'std time for N value {N_vals[i]} is {np.std(average_cos_times[:][i])}')




    CN_option_prices = np.array(CN_option_prices)
    FTCS_option_prices = np.array(FTCS_option_prices)
    CN_deltas = np.array(CN_deltas)/(S_M/M)
    FTCS_deltas = np.array(FTCS_deltas)/(S_M/M)
    all_cos_results = np.array(all_cos_results)
    d1,d2 = d1_d2(stock_price, params['K'][0], params['r'][0], params['sigma'][0], params['T'][0])

    option_prices = call_option_price(stock_price, params['K'][0], params['r'][0], params['T'][0], d1, d2)
    plt.plot(stock_price[1:-1], CN_deltas[-1,:], label='CN')
    plt.plot(stock_price[1:-1], FTCS_deltas[-1,:], label='FTCS')
    plt.plot(stock_price, bs_delta(d1), label='Black Scholes')
    plt.xlabel('Stock Price')
    plt.ylabel('Delta')
    plt.grid()
    plt.legend()
    plt.show()

    #Point Experiments
    d1,d2 = d1_d2(stock_price_points, params['K'][0], params['r'][0], params['sigma'][0], params['T'][0])
    all_cos_results = all_cos_results[:,:,0]
    option_price_points = call_option_price(stock_price_points, params['K'][0], params['r'][0], params['T'][0], d1, d2)
    # plt.scatter(stock_price_points, option_price_points, label='Black Scholes')
    # plt.scatter(stock_price_points, CN_option_prices[-1,stock_price_points], label='CN')
    # plt.scatter(stock_price_points, FTCS_option_prices[-1,stock_price_points], label='FTCS')
    # plt.scatter(stock_price_points, all_cos_results[:,2], label='Fourier-Cosine k=32')
    # plt.scatter(stock_price_points, all_cos_results[:,3], label='Fourier-Cosine k=64')
    # plt.scatter(stock_price_points, all_cos_results[:,4], label='Fourier-Cosine k=128')

    # plt.xlabel('Stock Price')
    # plt.ylabel('Option Price')
    # plt.grid()
    # plt.legend()
    # plt.show()

    error_CN = np.log(abs(option_price_points - CN_option_prices[-1,stock_price_points]))
    error_FTCS = np.log(abs(option_price_points - FTCS_option_prices[-1,stock_price_points]))
    error_cos_8 = np.log(abs(option_price_points - all_cos_results[:,0]))
    error_cos_16 = np.log(abs(option_price_points - all_cos_results[:,1]))
    error_cos_32 = np.log(abs(option_price_points - all_cos_results[:,2]))
    error_cos_64 = np.log(abs(option_price_points - all_cos_results[:,3]))
    error_cos_96 = np.log(abs(option_price_points - all_cos_results[:,4]))
    error_cos_128 = np.log(abs(option_price_points - all_cos_results[:,5]))
    error_cos_160 = np.log(abs(option_price_points - all_cos_results[:,6]))
    error_cos_192 = np.log(abs(option_price_points - all_cos_results[:,7]))
    print(f'CN errors is {error_CN}')
    print(f'FTCS errors is {error_FTCS}')
    print(f'Cosine errors N=8 is {error_cos_8}')
    print(f'Cosine errors N=16 is {error_cos_16}')
    print(f'Cosine errors N=32 is {error_cos_32}')
    print(f'Cosine errors N=64 is {error_cos_64}')
    print(f'Cosine errors N=96 is {error_cos_96}')
    print(f'Cosine errors N=128 is {error_cos_128}')
    print(f'Cosine errors N=160 is {error_cos_160}')
    print(f'Cosine errors N=192 is {error_cos_192}')


    plt.plot(stock_price_points, error_CN, label='CN')
    plt.plot(stock_price_points, error_FTCS, label='FTCS')

    plt.plot(stock_price_points, error_cos_64, label='Fourier-Cosine k=64')

    plt.xlabel('Stock Price')
    plt.ylabel('Log(Error)')
    plt.grid()
    plt.legend()
    plt.show()







    plt.plot(stock_price, option_prices, label='Black Scholes')
    plt.plot(stock_price, CN_option_prices[-1,:], label='CN')
    plt.plot(stock_price, FTCS_option_prices[-1,:], label='FTCS')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.grid()
    plt.legend()
    plt.show()

    # FTCS
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(projection='3d')


    stock_price_grid,  time_grid= np.meshgrid(stock_price, time_arr )

    surface = ax.plot_surface(stock_price_grid, time_grid, FTCS_option_prices, cmap='coolwarm')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('Option Price')
    ax.view_init(15, 250)
    ax.set_title('FTCS')
    plt.show()
    # CM
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(projection='3d')

    stock_price_grid,  time_grid= np.meshgrid(stock_price, time_arr )

    surface = ax.plot_surface(stock_price_grid, time_grid, CN_option_prices, cmap='coolwarm')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('Option Price')
    ax.set_title('Crank Nicolson')
    ax.view_init(15, 250)
    plt.show()


    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(projection='3d')

    stock_price_grid,  time_grid= np.meshgrid(stock_price, time_arr )

    surface = ax.plot_surface(stock_price_grid, time_grid, CN_option_prices-FTCS_option_prices, cmap='coolwarm')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('CM - FTCS Option Price Difference')
    ax.set_title('Difference')
    ax.view_init(15, 250)
    plt.show()

    # # print(FTCS_option_price-CN_option_price)
    # # plt.plot(stock_price, FTCS_option_price, label='FTCS')
    # # plt.plot(stock_price, CN_option_price, label='Crank-Nicolson', ls='--')
    # S_0 = np.linspace(0, M, M)
    # d1,d2 = d1_d2(stock_price, params['K'][0], params['r'][0], params['sigma'][0], params['T'][0])
    #
    # option_prices = call_option_price(stock_price, params['K'][0], params['r'][0], params['T'][0], d1, d2)
    # plt.plot(stock_price, option_prices, c='r')
    # plt.legend()
    # plt.show()

