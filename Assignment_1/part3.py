import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm

def d1_d2(S0, K, r, sigma, T):
    d1 = (log(S0/K) + (r+sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return d1,d2
def call_option_price(S0, K, r, T, d1, d2):
    return S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def put_option_price(S0, K, r, T, d1,d2):
    return K*exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
maturity = 365
S0 = 100
strike = 99
K = 100
volatility_delta = 0.2
volatility_stock = 0.2
r = 0.06
hedge_frequency = 7



def GBM_euler_mthod(S0, maturity, hedging_frequency):
    all_stock_prices = []
    all_call_option_prices = []
    all_call_options_hedged_daily = []
    all_put_option_prices = []
    all_deltas = []
    all_daily_deltas = []
    S = S0
    stock_ticks = int(maturity)
    for m in range(stock_ticks):
        norm_sampled = np.random.normal(0, 1)
        t = maturity - m

        S = S + r*S*stock_ticks**-1 + volatility_stock*S*sqrt(stock_ticks**-1)*norm_sampled
        d1, d2 = d1_d2(S, K, r, volatility_delta, t)
        normal_d1_daily = norm.cdf(d1)
        call_option = call_option_price(S, K, r, t, d1, d2)
        if m%hedging_frequency == 0 and hedging_frequency != 1:
            d1, d2 = d1_d2(S, K, r, volatility_delta, t)

            normal_d1 = norm.cdf(d1)
            all_deltas.append(normal_d1)
            call_option = call_option_price(S, K, r, t,d1,d2)
            put_option = put_option_price(S, K, r, t,d1,d2)


            all_put_option_prices.append(put_option)
            all_call_options_hedged_daily.append(call_option)
        all_call_option_prices.append(call_option)
        all_stock_prices.append(S)
        all_daily_deltas.append(normal_d1_daily)
    return all_stock_prices, all_call_option_prices, all_call_options_hedged_daily, all_put_option_prices, all_deltas, all_daily_deltas



def plotting_func(x,y, label = None):
    if label:
        plt.plot(x,y, label = label)
    else:

        plt.plot(x,y)



if __name__ == "__main__":
    stock_prices, call_option_prices, all_call_options_hedged_daily, put_option_prices, deltas, all_daily_deltas = GBM_euler_mthod(S0, maturity, hedge_frequency)
    deltas = [(all_daily_deltas, f'Delta Hedged Daily'),(deltas, f'Delta Hedged Every {hedge_frequency} days')]
    plotted_prices = [(stock_prices, 'Stock Price'), (call_option_prices, 'Option Price Hedged Daily'), (all_call_options_hedged_daily, f'Option Price Hedged Every {hedge_frequency} days')]#, (put_option_prices, 'Put Option Price') ]
    for i in plotted_prices:
        if i[1] == 'Stock Price':
            plotting_func(np.linspace(0,365, 365), i[0], i[1])
        else:
            plotting_func(np.linspace(0,365, len(i[0])), i[0], i[1])
    plt.title(f'Delta Volatility: {volatility_delta}, Stock Volatility: {volatility_stock}')
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()
    for i in deltas:
        plotting_func(np.linspace(0,365, len(i[0])), i[0], i[1])
    plt.title(f'Delta Volatility: {volatility_delta}, Stock Volatility: {volatility_stock}')
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Delta')
    plt.show()

