import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm
from mpl_toolkits import mplot3d
import tqdm

# Option pricing functions
def d1_d2(S0, K, r, sigma, T):
    d1 = (log(S0/K) + (r+sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return d1,d2
def call_option_price(S0, K, r, T, d1, d2):
    return S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def put_option_price(S0, K, r, T, d1,d2):
    return K*exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

# Hedging function
def GBM_euler_mthod(S0, K, r, volatility_delta, volatility_stock, T, M, hedge_frequency):
    all_stock_prices = []
    subsampled_stock_prices = []
    all_call_option_prices = []
    all_call_options_hedged_daily = []
    all_put_option_prices = []
    all_deltas = []
    all_daily_deltas = []
    dt = T/M
    daily_sampling = int(M/365)
    for m in range(M):
        # If init, we dont need to calculate price change
        if m == 0:
            t = T
            S = S0
        # Find Brownian Motion of the Underlying price
        else:
            norm_sampled = np.random.normal(0, 1)
            t = T-dt*m
            S = S + r*S*dt + volatility_stock*S*sqrt(dt)*norm_sampled
        if m%daily_sampling == 0:
            d1, d2 = d1_d2(S, K, r, volatility_delta, t)
            # Daily Hedging
            normal_d1_daily = norm.cdf(d1)
            call_option = call_option_price(S, K, r, t, d1, d2)
            all_daily_deltas.append(normal_d1_daily)
            all_call_option_prices.append(call_option)
            subsampled_stock_prices.append(S)

        # Custom Frequency hedging
        if m % int(M*hedge_frequency/365) == 0:
            d1, d2 = d1_d2(S, K, r, volatility_delta, t)

            normal_d1 = norm.cdf(d1)
            all_deltas.append(normal_d1)
            call_option = call_option_price(S, K, r, t,d1,d2)
            put_option = put_option_price(S, K, r, t,d1,d2)


            all_put_option_prices.append(put_option)
            all_call_options_hedged_daily.append(call_option)
        all_stock_prices.append(S)

    return all_stock_prices,subsampled_stock_prices,  all_call_option_prices, all_call_options_hedged_daily, all_put_option_prices, all_deltas, all_daily_deltas

# Plotting function
def plotting_func(x,y, label = None):
    if label != 'Stock Price':
        plt.step(x,y, where='post',label = label)
    elif label == 'Stock Price':
        plt.plot(x,y, label = label)
    else:

        plt.plot(x,y)

# Default option parameters
T = 1.0
S0 = 100
K = 99      # Strike price
volatility_delta = 0.2
volatility_stock = 0.2
r = 0.06
hedge_frequency = 7     # Weekly (given in amount of days)
M = 3650                # Discretization resolution

if __name__ == "__main__":
    stock_prices, subsampled_stock_prices, call_option_prices, all_call_options_hedged_weekly, put_option_prices, deltas, all_daily_deltas = GBM_euler_mthod(S0, K, r, volatility_delta, volatility_stock, T, M, hedge_frequency)
    deltas = [(all_daily_deltas, f'Delta Hedged Daily'),(deltas, f'Delta Hedged Every {hedge_frequency} days')]
    plotted_prices = [(stock_prices, 'Stock Price'), (call_option_prices, 'Option Price Hedged Daily'), (all_call_options_hedged_weekly, f'Option Price Hedged Every {hedge_frequency} days'), (put_option_prices, 'Put Option Price') ]
    for i in plotted_prices:
        if i[1] == 'Stock Price':
            plotting_func(np.linspace(0, M, M), i[0], i[1])
        else:
            plotting_func(np.linspace(0, M, len(i[0])), i[0], i[1])
    plt.title(f'Delta Volatility: {volatility_delta}, Stock Volatility: {volatility_stock}')
    plt.legend()
    plt.grid()
    plt.xlabel('Discretization Steps')
    plt.ylabel('Stock Price')
    plt.show()
    for i in deltas:
        plotting_func(np.linspace(0, M, len(i[0])), i[0], i[1])
    plt.title(f'Delta Volatility: {volatility_delta}, Stock Volatility: {volatility_stock}')
    plt.legend()
    plt.grid()
    plt.xlabel('Discretization Steps')
    plt.ylabel('Delta')
    plt.show()

    # volatilities = [0.1,0.2,0.3,0.4,0.5]
    volatilities = [0.2, 0.9]
    for volatlity in volatilities:
        all_stock_prices = []
        all_call_option_prices = []
        all_deltas = []
        all_times = []
        for i in tqdm.tqdm(range(100)):
            stock_prices, subsampled_stock_prices,call_option_prices, all_call_options_hedged_weekly, put_option_prices, deltas, all_daily_deltas = GBM_euler_mthod(
                S0, K, r, volatility_delta, volatlity, T, M, hedge_frequency)

            all_stock_prices.append(subsampled_stock_prices)
            all_call_option_prices.append(call_option_prices)
            all_deltas.append(all_daily_deltas)
            all_times.append(list(np.linspace(0, 365, 365)))

        combined_stock_prices = sum(all_stock_prices, [])
        combined_call_option_prices = sum(all_call_option_prices, [])
        combined_deltas = sum(all_deltas, [])
        combined_times = sum(all_times, [])
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        # Creating plot
        plot = ax.scatter3D( combined_stock_prices, combined_times, combined_call_option_prices,c=combined_deltas)
        plt.title(f"Change in Delta over Time wrt Stock Price \n Delta Volatility: {volatility_delta}, Stock Volatility: {volatlity}")
        ax.set_xlabel('Stock Price', fontweight='bold')
        ax.set_ylabel('Time', fontweight='bold')
        ax.set_zlabel('Option Price', fontweight='bold')
        fig.colorbar(plot, ax=ax, shrink=0.5, aspect=5, label = 'Delta')
        ax.view_init(15, -115)
        # plt.scatter(combined_stock_prices, combined_call_option_prices, c=combined_deltas, alpha=0.5)
        #
        # plt.xlabel('Stock Price')
        # plt.ylabel('Option Price')
        # plt.colorbar(label='Delta')
        # plt.grid()
        # plt.title(f'Effect of Option and Stock Price on Delta with \n Delta Volatility: {volatlity}, Stock Volatility: {volatility_stock}')
        # plt.legend()
        plt.show()
    test = 0

