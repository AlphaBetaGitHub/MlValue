from pathlib import Path

from portfolio_metrics import portfolio_metrics, yearly_returns
import pandas as pd

from alpha_go.consts import LOCAL_PATH
import pdb

class ReportBuilder(object):
    def __init__(self):
        self.returns = {}
    def set_args(self, year, predictions, portfolio_size, model_type):
        self.year = year
        self.y_pred = predictions.y_pred
        self.output = predictions.reset_index()
        self.portfolio_size = portfolio_size
        self.model_type = model_type
        self.output_folder = LOCAL_PATH+f'/results/{model_type}/'
        print(f'{year} {model_type}')

    def calc_portfolio_returns(self):
        """
        calculate the returns for the predictions
        :return: None
        """

        # sort values by date and prediction
        self.predictions = self.output.sort_values(['date', 'y_pred']).reset_index(drop=True)

        # rank prediction by date and return
        self.predictions['rank'] = self.predictions.groupby(['date', 'y_pred']).ngroup()
        self.predictions['rank'] = self.predictions.groupby('date')['rank'].rank()

        # take the top and bottom index
        top_index = self.predictions.groupby('date')['rank'].nlargest(self.portfolio_size).index.levels[1]
        btm_index = self.predictions.groupby('date')['rank'].nsmallest(self.portfolio_size).index.levels[1]
        self.top_index = top_index
        self.btm_index = btm_index
        # check overlapping stocks between top and btm
        intersection = set(top_index).intersection(set(btm_index))
        if len(intersection) > 0:
            print('The top and the btm portfolio share the same stocks')
            print(self.predictions[self.predictions.index.isin(intersection)])
        
        # take the top and btm returns
        #pdb.set_trace() 
        #print(self.predictions.iloc[top_index])
        #print(self.predictions.iloc[btm_index])
        self.top_returns = self.predictions[self.predictions.index.isin(top_index)].groupby('date')['y_test'].mean()
        self.btm_returns = self.predictions[self.predictions.index.isin(btm_index)].groupby('date')['y_test'].mean()
        self.top_btm_returns = self.top_returns - self.btm_returns
        self.returns.setdefault('top',[]).append(self.top_returns)
        self.returns.setdefault('btm',[]).append(self.btm_returns)
        self.returns.setdefault('top_btm',[]).append(self.top_btm_returns)
    def calc_portfolio_metrics(self):
        """
        calculate the portfolio returns
        :return: None
        """

        # calc top and btm portfolio metrics
        top_metrics = portfolio_metrics(pd.concat(self.returns['top']), self.model_type, f'TOP {self.portfolio_size}')
        btm_metrics = portfolio_metrics(pd.concat(self.returns['btm']), self.model_type, f'BTM {self.portfolio_size}')
        top_btm_retruns = portfolio_metrics(pd.concat(self.returns['top_btm']), self.model_type,
                                            f'TOP {self.portfolio_size}-BTM {self.portfolio_size}')
        self.portfolio_metrics = pd.concat([top_metrics, btm_metrics, top_btm_retruns])

        # calc top and btm yearly returns
        top_yearly = yearly_returns(pd.concat(self.returns['top']), self.model_type, f'TOP {self.portfolio_size}')
        btm_yearly = yearly_returns(pd.concat(self.returns['btm']), self.model_type, f'BTM {self.portfolio_size}')
        top_btm_yearly = yearly_returns(pd.concat(self.returns['top_btm']), self.model_type,
                                        f'TOP {self.portfolio_size}-BTM {self.portfolio_size}')
        self.yearly_returns = pd.concat([top_yearly, btm_yearly, top_btm_yearly]).round(1)

    def store_backtest(self):
        """
        save portfolio returns
        :return: None
        """
        writer = pd.ExcelWriter(Path(self.output_folder + self.model_type + f'_metrics_{self.year}' + '.xlsx'), engine='openpyxl')
        self.portfolio_metrics.to_excel(writer, sheet_name=f'portfolio_metrics_{self.portfolio_size}_{self.year}')
        self.yearly_returns.to_excel(writer, sheet_name=f'yearly_returns_{self.portfolio_size}_{self.year}')
        self.portfolio_metrics.to_csv(Path(self.output_folder+self.model_type+f'_metrics_pf{self.portfolio_size}_{self.year}.csv'))
        self.yearly_returns.to_csv(Path(self.output_folder+self.model_type+f'_yearly_pf{self.portfolio_size}_{self.year}.csv'))
        writer.save()

    def run(self, year, predictions, portfolio_size, model_type):
        self.set_args(year, predictions, portfolio_size, model_type)
        self.calc_portfolio_returns()
        self.calc_portfolio_metrics()
        print(self.portfolio_metrics)
        print(self.yearly_returns)
        self.store_backtest()
        return {'top' : self.predictions[self.predictions.index.isin(self.top_index)],
                'btm' : self.predictions[self.predictions.index.isin(self.btm_index)]}
