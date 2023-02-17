import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm

def d1_d2(S0, K, r, sigma, T):
    d1 = (log(S0/K) + (r+sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return d1,d2
def call_option_price(S0, K, r, T, d1,d2):
    return S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
maturity = 365
S0 = 100
strike = 99
K = 100
volatility_delta = 0.9
volatility_stock = 0.2
r = 0.06
hedge_frequency = 1



def GBM_euler_mthod(S0, maturity, hedging_frequency):
    all_stock_prices = []
    all_call_option_prices = []
    all_deltas = []
    S = S0
    ticks = int(maturity/hedging_frequency)
    for m in range(ticks):
        norm_sampled = np.random.normal(0, 1)
        t = maturity - m * hedging_frequency
        d1, d2 = d1_d2(S, K, r, volatility_delta, t)
        S = S + r*S*ticks**-1 + volatility_stock*S*sqrt(ticks**-1)*norm_sampled


        normal_d1 = norm.cdf(d1)
        all_deltas.append(normal_d1)
        call_option = call_option_price(S, K, r, t,d1,d2)
        all_call_option_prices.append(call_option)
        all_stock_prices.append(S)
    return all_stock_prices, all_call_option_prices, all_deltas



def plotting_func(x,y, label = None):
    if label:
        plt.plot(x,y, label = label)
    else:

        plt.plot(x,y)



if __name__ == "__main__":
    stock_prices, option_prices, deltas = GBM_euler_mthod(S0, maturity, hedge_frequency)
    deltas = (deltas, 'Delta')
    plotted_prices = [(stock_prices, 'Stock Price'), (option_prices, 'Option Price')]
    for i in plotted_prices:
        plotting_func(np.arange(0,maturity,hedge_frequency), i[0], i[1])
    plt.title('Stock Price vs Option Price')
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()

    plotting_func(np.arange(0,maturity,hedge_frequency), deltas[0], deltas[1])
    plt.title('Delta vs Time')
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Delta')
    plt.show()
