# Portfolio-Builder

An object that takes daily stock data from NSDQ (Yahoo Finance), and builds a recommended portfolio (investment list)- based on the Universal Portfolio Algorithm, or the Multiplicative Updates algorithm

The PortfolioBuilder class contains the following methods:

• get_daily_data - The method access a public API and gets end-of-day stocks data.

• find_universal_portfolio – The method decides on a portfolio for the next day according to previously retrieved data. The method uses the universal portfolio algorithm.

• find_exponential_gradient_portfolio - The method decides on a portfolio for the next day according to previously retrieved data. The method uses the exponential gradient algorithm.
