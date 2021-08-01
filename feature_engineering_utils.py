import numpy as np
import pandas as pd


class FeatureConstructor(object):

    def __init__(self, all_data):
        self.all_data = all_data

    def momentum(self, mom_time):
        """
        Momentum with look back period
        :param mom_time: the time to look back
        :return: pre defined months momentum
        """
        self.all_data[f'mom_{mom_time}'] = (self.all_data.groupby("isin")["price_close_adj"].shift(1) /
                                  self.all_data.groupby("isin")["price_close_adj"].shift(mom_time + 1) - 1)

    def log_market_cap(self):
        self.all_data['logsize'] = np.log(self.all_data['price_close_adj'] * self.all_data['shares'])

    def change_in_six_month_mom(self):
        """
        :return: the change in the six month momentum feature
        """
        self.all_data['chmom_6'] = (self.all_data.groupby("isin")["mom_6"].pct_change(1))

    @staticmethod
    def max_daily_return():
        print('Max daily return feature constructed in preparation part')
        pass

    def industry_twelve_month_mom(self):
        """
        :return: twelve month industry momentum with 1 month look out
        """
        self.all_data['indmom_a_12'] = self.all_data.groupby(['month','simpleindustryid'])['mom_12'].transform('mean')

    @staticmethod
    def return_volatility():
        print('return volatility feature constructed in preparation part')
        pass

    def log_dollar_trading_volume(self):
        self.all_data['logdolvol'] = np.log(self.all_data['price_close_adj'] * self.all_data['volume'])

    def sales_to_price(self):
        self.all_data['sp'] = self.all_data.price_close_adj / self.all_data.revenue

    def share_turnover(self):
        self.all_data['turn'] = self.all_data.volume / self.all_data.shares

    def book_to_market(self):
        self.all_data['b2m'] = self.all_data.book_value / self.all_data.price_close_adj

    def construct_all_features(self):
        self.momentum(mom_time=1)
        self.log_market_cap()
        self.momentum(mom_time=12)
        self.momentum(mom_time=6)
        self.change_in_six_month_mom()
        self.max_daily_return()
        self.industry_twelve_month_mom()
        self.return_volatility()
        self.log_dollar_trading_volume()
        self.sales_to_price()
        self.share_turnover()
        self.book_to_market()

        return self.all_data


class SklearnWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, df):
        transformed = self.transform.fit_transform(df.values)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index)
