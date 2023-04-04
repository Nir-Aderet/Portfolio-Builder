import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date
import itertools
import math
import yfinance
from typing import List


class PortfolioBuilder:

    def __init__(self):
        self.data = ''
        self.data_ar = ''
        self.x = []
        self.list_of_stacks = []
        self.b0 = ''
        self.time_index = []
        self.all_b_g = []
        self.all_b_u = []
        pass

    def get_daily_data(self, tickers_list: List[str],
                       start_date: date,
                       end_date: date = date.today()
                       ) -> pd.DataFrame:
        """
        get stock tickers adj_close price for specified dates.

        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame

        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """'''
        f = web.DataReader(tickers_list, 'yahoo', start_date, end_date)
        f_second = f["Adj Close"]
        data_frame = pd.DataFrame(f_second)
        return data_frame
        '''
        try:
            f = web.DataReader(tickers_list, 'yahoo', start_date, end_date)
            f_second = f["Adj Close"]
            self.b0 = np.zeros(len(tickers_list))
            self.b0.fill(1 / len(tickers_list))  # default b Zero
            self.list_of_stacks = tickers_list
            self.data = f_second  # dataframe type of stock prices
            self.data_ar = f_second.to_numpy()  # array type of stock prices
            self.time_index = self.data.index.to_numpy()  # list of dates with prices
            self.all_b_g.append(self.b0)  # list of b for exponential gradient
            self.all_b_u.append(self.b0)  # list of b for universal portfolio
            for m in range(1, len(self.time_index)):  # create the X's for the given data
                list = []
                for j in range(len(self.list_of_stacks)):
                    list.append(self.data_ar[m, j] / self.data_ar[m - 1, j])
                self.x.append(list)
            return f_second
            pass
        except Exception:
            raise ValueError

    def rec_permutation(self, data, perm_length):
        """
        This function is a recursive implementation of choosing "k" elements
        out of "n" options with returning occurrences
        :param data: the "n" length data base
        :param perm_length: the number of elements to choose "k"
        :return: res_list: a list of lists of all possible permutations
        """
        if perm_length == 1:
            return [[atom] for atom in data]

        res_list = []
        smaller_perm = self.rec_permutation(data, perm_length - 1)
        for elem in data:
            for sub_combination in smaller_perm:
                res_list.append([elem] + sub_combination)
        return res_list

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """
        sum1_w = []  # list of all the possibilities that their sum is 1
        all_w = self.rec_permutation(
            np.arange(0.0, 1.0 + 1 / float(portfolio_quantization), 1 / float(portfolio_quantization)).tolist(),
            len(self.list_of_stacks))
        for i in range(len(all_w)):  # check all the possibilities for a possibility with a sum of 1
            sum = 0  # initializing a counter to check the sum
            for j in range(len(self.list_of_stacks)):  # check the sum of a possibility
                sum += all_w[i][j]
            if 1.0009 > sum > 0.9995:
                sum1_w.append(all_w[i])
        wealth = [1.0]  # initializing a counter to register the wealth
        s = 1  # np.dot(self.all_b_u[0], self.x[0])
        sum1_w = np.array(sum1_w)
        for m in range(1, len(self.time_index)):  # day
            sum, sum_down = 0, 0
            for j in range(len(sum1_w)):  # go over all the possibilities in omega
                st = 1
                for k in range(m):  # go over all previous days
                    st *= np.dot(sum1_w[j], self.x[k])  # compute the wealth with the same option from omega
                sum_down += st
                yossi = (sum1_w[j] * st)
                sum = sum + yossi
            self.all_b_u.append(sum / sum_down)  # add the given b to a list of all the b
            s *= np.dot(self.all_b_u[m - 1], self.x[m - 1])
            wealth.append(s)
        self.all_b_g = [self.b0]
        return wealth
        pass

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        """
        calculates the exponential gradient portfolio for the previously requested stocks

        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: returns a list of floats, representing the growth trading  per day
        """

        wealth = [1.0]
        s = 1
        for m in range(1, len(self.time_index)):  # day
            b = []
            sum = 0
            for j in range(len(self.list_of_stacks)):
                nominator = self.all_b_g[m - 1][j] * np.exp(
                    (learn_rate * self.x[m - 1][j]) / np.dot(self.all_b_g[m - 1], self.x[m - 1]))
                sum += nominator
            for j in range(len(self.list_of_stacks)):
                up = self.all_b_g[m - 1][j] * np.exp(
                    (learn_rate * self.x[m - 1][j]) / np.dot(self.all_b_g[m - 1], self.x[m - 1]))
                b.append(up / sum)
            self.all_b_g.append(b)
            s *= np.dot(self.all_b_g[m - 1], self.x[m - 1])
            wealth.append(s)
        self.all_b_g = [self.b0]
        return wealth

        pass


if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    print('write your tests here')  # don't forget to indent your code here!
    f = PortfolioBuilder()
    start = date(2020, 1, 1)
    end = date(2020, 2, 1)
    g = f.get_daily_data(['GOOG', 'AAPL', 'MSFT'], start, end)
    print(g)
    print(f.find_exponential_gradient_portfolio())
    print(f.find_universal_portfolio(20))
