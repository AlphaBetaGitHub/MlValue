LOCAL_PATH = './data/'

UNIVERSE_FILE_NAME = 'sw_universe.csv'

COUNTRY_CODE = 194

BOOK_VALUE_CODE = 4020
REVENUE_CODE = 112
SHARES_CODE = 1070


'''
DATA_STYLE can be one of the following:
1. ALPHAGO_PAPER - to get data from the DB for the factors used in Alpha Go
   Everywhere paper. The processed dataframe will be pickled to DATA_SRC
2. UNPROCESSED_CSV - to get data from a CSV file
3. PROCESSED_PICKLE - a processed pickle file. See DATA_FIELDS for details
'''


DATA_STYLE = 'UNPROCESSED_CSV'

# Path for the data file and the data separator used
DATA_SRC = 'data/america_factors_constituents.txt'
DATA_SEP = '\t'
COUNTRY = 'UNITED STATES'

'''
DATA_FIELDS specifies the input data fields to be used. In the case of processed
pickle files, the first two fields should be "date" and "isin" and the last two
fields should be "pct_change" (for the universe_returns, i.e. the target variable) 
and "backtest returns" i.e. the returns used for evaluation in the backtest. The 
fields in between are the factors that needs to be used. The number of such 
factors should be specified in NUM_FEAT.
'''
DATA_FIELDS = ['Periods', 'isin','Size',
       'Beta3Y', 'Beta5Y', 'LowVol1M', 'LowVol12M', 'MAX', 'MAX5',
       'MOMENTUM6M', 'MOMENTUM9M', 'MOMENTUM12M', 'VAR', 'MAD', 'ROE', 'ROA',
       'ROIC', 'ROC', 'CBGPR', 'CBOPR', 'GPR', 'OPR', 'C2D', 'E2M', 'C2M',
       'B2M', 'CBGPR2M', 'CBOPR2M', 'GPR2M', 'OPR2M', 'SHAREVOL', 'STDTURN',
       'BOLLINGERLOWERBAND', 'SHAREGROWTH1Y', 'COMPOSITEISSUANCE',
       'EBITDA-TO-EV', 'EBIT-TO-EV', 'NETEQUITYFINANCE', 'XFIN', 'DivYield', 'Universe_Returns','Backtest Returns']

# The number of factors
NUM_FEAT = 38

'''
When the names of the first two and last two input fields are different from 
what is specified above, specify how they needs to be renamed
'''
RENAME_FIELDS = {'Periods': 'date', 'Backtest Returns': 'backtest returns', 'Universe_Returns': 'pct_change', 'Country': 'country'}

# Date format
DATE_FORMAT = "%Y%m%d"

# NN consts
EPOCHS_NUMBER = 100
BATCH_SIZE = 1024
#for all stocks
LR = 0.1
PENALTY = 0.001
PATIENCE = 15
L1 = 0.001

''' 
The normalization used for the features can be
1. RANK for Kelly style normalization by assigning ranks based on each factor
and then transforming the ranks to the range -1 to 1.
2. ZSCORE for transforming the features using z-score
'''
FEAT_NORM = 'RANK'

# validation size used only in the case of hyper parameter estimation. 
# For NN training, the validation size is set to last 6 months.
VALIDATION_SIZE = 12 * 2


# MODE can be 'REG' for regression training or 'CLS' for classification.
MODE = 'REG'

PORTFOLIO_SIZE = 50  # could be 20

'''
Options to fill missing values. Fields specified in FILLNA_ZERO_FIELDS are 
filled with zeros for missing values. The rest are filled with cross-section
median if FILLNA_MEDIAN is set to True.
''' 
FILLNA_ZERO_FIELDS = ('SHAREGROWTH1Y', 'DivYield')
FILLNA_MEDIAN = True

