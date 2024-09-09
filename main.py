import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date
import itertools
import math
import yfinance
from typing import List
import datetime


class PortfolioBuilder:

    def get_daily_data(self, tickers_list: List[str], start_date: date, end_date: date = date.today()) -> pd.DataFrame:
        """
       get stock tickers adj_close price for specified dates.

       :param List[str] tickers_list: stock tickers names as a list of strings.
       :param date start_date: first date for query
       :param date end_date: optional, last date for query, if not used assumes today
       :return: daily adjusted close price data as a pandas DataFrame
       :rtype: pd.DataFrame

       example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
       """
        try:
            df = web.get_data_yahoo(tickers_list, start_date, end_date)
            self.data_frame = df["Adj Close"]
            self.tickers_list = tickers_list
            self.start_date = start_date
            self.end_date = end_date
            end_date = end_date + datetime.timedelta(days=1)
            if self.data_frame.isnull().values.any():
                raise ValueError
            return self.data_frame
        except Exception:
            raise ValueError

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """
        x_vec = []
        for t in range(len(self.data_frame) - 1):
            x_vec.append(self.vector_x(t + 1))
        x_vec = np.array(x_vec)
        all_options = self.bw(portfolio_quantization)
        b1 = np.zeros(len(self.tickers_list))
        b1 += (1 / len(self.tickers_list))
        bw = [i for i in all_options if 0.999 <= np.sum(i) <= 1.001]
        b_vec = np.zeros((len(self.data_frame), len(self.tickers_list)))
        b_vec[0] = b1
        s = np.zeros(len(self.data_frame))
        s[0] = 1
        for day in range(1, len(self.data_frame)):
            if day == 1:
                s[1] = np.dot(b1, x_vec[0])
                result = [np.dot(a, x_vec[day - 1]) for a in bw]
                denominator = np.sum(result)
                sum2 = [a * j for a, j in zip(bw, result)]
                numerator = np.sum(sum2, axis=0)
                b_vec[day] = numerator / denominator
            else:
                s[day] = np.dot(s[day - 1], np.dot(b_vec[day - 1], x_vec[day - 1]))
                s_lis = []
                result = []
                for i in bw:
                    s_list = []
                    for j in range(0, day):
                        s_list.append(np.dot(x_vec[j], i))
                    s_lis.append(np.prod(s_list))
                    u_sum = i * np.prod(s_list)
                    result.append(u_sum)
                denominator = np.sum(s_lis)
                numerator = np.sum(result, axis=0)
                b_vec[day] = numerator / denominator
        return s

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        """
        calculates the exponential gradient portfolio for the previously requested stocks

        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: returns a list of floats, representing the growth trading  per day
        """
        s = [float(1)]
        for t in range(len(self.data_frame) - 1):
            b = self.exponential_gradient(t, learn_rate)
            x_dot_b = np.array(b).dot(np.array(self.vector_x(t + 1)))
            s.append(s[t] * x_dot_b)
        return s

    def vector_x(self, t):
        df = self.data_frame
        currentX = df.iloc[t]
        previousX = df.iloc[t - 1]
        self.x = [numerator / denominator for numerator, denominator in zip(currentX, previousX)]
        return self.x

    def exponential_gradient(self, t, n: float = 0.5):
        t += 1
        if t == 1:
            self.bt = np.ones(len(self.tickers_list)) / len(self.tickers_list)
            return self.bt
        else:
            saving_data = []
            new_bt = []
            for i in range(len(self.tickers_list)):
                x = self.vector_x(t - 1)
                bti = self.bt[i]
                x_array = np.array(x)
                bt_array = np.array(self.bt)
                bt = bti * math.exp((n * x[i]) / bt_array.dot(x_array))
                saving_data.append(bt)
            for j in range(len(self.tickers_list)):
                bj = saving_data[j] / sum(saving_data)
                new_bt.append(bj)
            self.bt = np.array(new_bt)
        return self.bt

    def b_by_universal_portfolio(self, t, a):
        if t == 1:
            self.bt = np.ones(len(self.tickers_list)) / len(self.tickers_list)
            return self.bt
        else:
            denominator = 0
            numerator = 0
            for option in self.bw(a):
                bw_dot_x = 1
                for day in range(1, t):
                    x = self.vector_x(day)
                    s = np.array(option).dot(self.vector_x(day))
                    bw_dot_x = bw_dot_x * np.array(option).dot(self.vector_x(day))
                s = bw_dot_x
                s_dot_option = np.array(s).dot(np.array(option))
                denominator += bw_dot_x
                numerator += s_dot_option
            b = numerator / denominator
        return b

    def bw(self, portfolio_quantization):
        bw = np.arange(0.0, 1.0, (1 / portfolio_quantization))
        bw = np.append(1.0, bw)
        product = list(itertools.product(bw, repeat=len(self.tickers_list)))
        all_options = [np.round(elem, 10) for elem in product]
        return all_options


if __name__ == '__main__': .
    print('write your tests here')!
