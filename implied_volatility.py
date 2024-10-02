import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import pandas as pd
import pandas_datareader.data as web
import datetime as dt

N = norm.cdf

# S : current asset price
# K: strike price of the option
# r: risk free rate
# T : time until option expiration
# σ: annualized volatility of the asset's returns


def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * N(d1) - K * np.exp(-r*T)* N(d2)


def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)


def implied_vol(opt_value, S, K, T, r, type_):
    try:
        def call_obj(sigma):
            return abs(BS_CALL(S, K, T, r, sigma) - opt_value)

        def put_obj(sigma):
            return abs(BS_PUT(S, K, T, r, sigma) - opt_value)

        if type_ == 'call':
            res = minimize_scalar(call_obj, bounds=(0.01, 6), method='bounded')
            return res.x
        elif type_ == 'put':
            res = minimize_scalar(put_obj, bounds=(0.01, 6), method='bounded')
            return res.x
        else:
            raise ValueError("type_ must be 'put' or 'call'")
    except Exception:
        raise


def main():
    try:
        # Get user input
        nifty_ce_price = float(input("Enter the Nifty CE price: "))
        nifty_pe_price = float(input("Enter the Nifty PE price: "))
        spot_price = float(input("Enter the spot price: "))
        strike_price = float(input("Enter the strike price: "))
        time_to_expire = float(input("Enter the time to expire (in days): "))
        risk_free_rate = float(input("Enter the risk-free rate (in percentage): "))

        # Calculate call and put implied volatility
        call_iv = implied_vol(
            nifty_ce_price,
            spot_price,
            strike_price,
            time_to_expire,
            risk_free_rate,
            'call',
        )

        put_iv = implied_vol(
            nifty_pe_price,
            spot_price,
            strike_price,
            time_to_expire,
            risk_free_rate,
            'put',
        )

        # Print results
        print(f'CE IV: {call_iv}')
        print(f'PE IV: {put_iv}')
    
    except ValueError:
        print("Please enter valid numerical values.")

# Assuming 'implied_vol' is already defined somewhere in your code



if __name__ == '__main__':
    main()
