from typing import Any
import pandas as pd
import numpy as np

import math
import yfinance as yf

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

from sklearn.svm import SVR

import warnings
warnings.filterwarnings(action="ignore")

class TechnicalIndicators:
    def __init__(self, DF_INDEX, RSI_WINDOW_LENGTH, RV_WINDOW_LENGTH):
        self.df_index = DF_INDEX
        self.RSI_window_length = RSI_WINDOW_LENGTH
        self.RV_window_length = RV_WINDOW_LENGTH
            
    # Historically Realized Volatility based on 21 trading days
    def realized_volatility(self):
         
         Reailized_Volatility =  self.df_index['Close'].pct_change().rolling(window = self.RV_window_length).std(ddof = 0)*np.sqrt(252)
         return Reailized_Volatility

    # MACD
    def MACD(self):

        emv_12 =  self.df_index['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        emv_26 =  self.df_index['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        MACD = emv_12 - emv_26
        return MACD

    # RSI    
    def RSI(self):
        
        diff = self.df_index['Close'].diff(1)
        gain = diff.clip(lower=0).round(2)
        loss = diff.clip(upper=0).abs().round(2)

        avg_gain = gain.rolling(self.RSI_window_length).mean()
        avg_loss = loss.rolling(self.RSI_window_length).mean()

        RS = avg_gain / avg_loss

        RSI = 100 - (100 / (1 + RS))

        return RSI
    
class ModelTunning:
    def __init__(self, INDEX_SYMBOL, START, END, TRAINING_BARS, VALIDATION_BARS, TESTING_BARS, SCALER_X, SCALER_Y, CONFIG, TRANSACTION_COST):

        self.index_symbol = INDEX_SYMBOL
        self.start = START
        self.end = END
    
        self.training_bars = TRAINING_BARS
        self.validation_bars = VALIDATION_BARS
        self.testing_bars = TESTING_BARS
        
        self.scaler_X = SCALER_X
        self.scaler_Y = SCALER_Y

        self.config = CONFIG
        self.transaction_cost = TRANSACTION_COST

    def get_data(self):
        df_index = yf.download(self.index_symbol, self.start, self.end).drop(columns=['Adj Close'])
        return df_index
    
       # seperation of lag data set 
    def create_dataset(self, X, y, time_steps):

        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values.flatten()

            Xs.append(v)        
            ys.append(y.iloc[i + time_steps])
            
        return np.array(Xs), np.array(ys)
    
    # splitting, adding features, scaling, and creating the dataset
    def data_preperation(self, df, ranges, num_lags, RSI_window, i, RV_window):

        ranges = list(range(self.training_bars, len(df) - self.testing_bars, self.validation_bars))

        train_start = max(0, ranges[i] - self.training_bars - 25 - num_lags)  
        train_end = ranges[i]
        training_data = df[train_start:train_end]

        val_start = ranges[i] - 25 - num_lags
        val_end = ranges[i] + self.validation_bars
        validation_data = df[val_start:val_end]

        test_start = ranges[i] + self.validation_bars - 25 - num_lags
        test_end = ranges[i] + self.validation_bars + self.testing_bars
        testing_data = df[test_start:test_end]

        training_data['MACD'] = TechnicalIndicators(training_data, RSI_window, RV_window).MACD()
        validation_data['MACD'] = TechnicalIndicators(validation_data, RSI_window, RV_window).MACD()
        testing_data['MACD'] = TechnicalIndicators(testing_data, RSI_window, RV_window).MACD()

        training_data['realized_volatility'] = TechnicalIndicators(training_data, RSI_window, RV_window).realized_volatility()
        validation_data['realized_volatility'] = TechnicalIndicators(validation_data, RSI_window, RV_window).realized_volatility()
        testing_data['realized_volatility'] = TechnicalIndicators(testing_data, RSI_window, RV_window).realized_volatility()

        training_data['RSI'] = TechnicalIndicators(training_data, RSI_window, RV_window).realized_volatility()
        validation_data['RSI'] = TechnicalIndicators(validation_data, RSI_window, RV_window).realized_volatility()
        testing_data['RSI'] = TechnicalIndicators(testing_data, RSI_window, RV_window).realized_volatility()

        training_data.dropna(inplace=True)
        validation_data.dropna(inplace=True)
        testing_data.dropna(inplace=True)

        train_X, train_Y = training_data.values, training_data['Close'].values
        val_X, val_Y = validation_data.values, validation_data['Close'].values
        test_X, test_Y = testing_data.values, testing_data['Close'].values

        if i==0:
            SC_train_X, SC_train_Y = self.scaler_X.fit_transform(train_X), self.scaler_Y.fit_transform(train_Y.reshape(-1,1))
        else:
            SC_train_X, SC_train_Y = self.scaler_X.transform(train_X), self.scaler_Y.transform(train_Y.reshape(-1,1))
        SC_val_X, SC_val_Y = self.scaler_X.transform(val_X), self.scaler_Y.transform(val_Y.reshape(-1,1))
        SC_test_X, SC_test_Y = self.scaler_X.transform(test_X), self.scaler_Y.transform(test_Y.reshape(-1,1))

        df_SC_train_X, df_SC_train_Y = pd.DataFrame(SC_train_X), pd.DataFrame(SC_train_Y)
        df_SC_val_X, df_SC_val_Y = pd.DataFrame(SC_val_X), pd.DataFrame(SC_val_Y)
        df_SC_test_X, df_SC_test_Y = pd.DataFrame(SC_test_X), pd.DataFrame(SC_test_Y)

        train_X, train_Y = self.create_dataset(df_SC_train_X, df_SC_train_Y, num_lags)
        val_X, val_Y = self.create_dataset(df_SC_val_X, df_SC_val_Y, num_lags)
        test_X, test_Y = self.create_dataset(df_SC_test_X, df_SC_test_Y, num_lags)

        return train_X, train_Y, val_X, val_Y, test_X, test_Y
    
    # creating one-step predictions
    def create_one_step_predictions(self, combined_train_val_X, test_X, model_test):
        history = combined_train_val_X # temporary name until i dont create more linear regressions.
        N_test_observations = len(test_X)

        model_predictions = np.zeros((0,1)).reshape(-1,1)
        for time_point in range(N_test_observations):
            y_hat = model_test.predict(history[-1:])
            model_predictions = np.concatenate((model_predictions, y_hat))

            real_test_value = test_X[time_point].reshape(1,-1)
            history = np.concatenate((history, real_test_value))

        test_predictions = self.scaler_Y.inverse_transform(model_predictions)

        return test_predictions

    def hyperparameter_tunning(self):

        df = self.get_data()

        ALL_MODEL_PREDICTIONS = np.zeros((0,1)).reshape(-1,1)
        ACTUAL_VALUES = np.zeros((0,1)).reshape(-1,1)

        ranges = list(range(self.training_bars, len(df) - self.testing_bars, self.validation_bars))
        for i in range(0, len(ranges)):
            
            df_finding_the_best_configuration = None

            training_IR2 = []
            validation_IR2 = []
            testing_IR2 = []
            config_all = []
    
            for cfg in self.config:
                train_X, train_Y, val_X, val_Y, test_X, test_Y = self.data_preperation(df, ranges, cfg['num_lags'], cfg['RSI_window'], i, cfg['RV_window'])

                # Fit the model and Predicting the **TRAINING** data
                model_train = LinearRegression().fit(train_X, train_Y)
                train_predictions = self.create_one_step_predictions(train_X, train_X, model_train)
                train_inversed = self.scaler_Y.inverse_transform(train_Y)

                df_train_predictions = pd.DataFrame(data={'Close': train_inversed.flatten(), 'Predictions': train_predictions.flatten()})
                df_train_predictions = BacktestStrategy(None, df_train_predictions, self.transaction_cost).backtesting()

                Train_IR2 = PerformanceMetrics(None, None, None, None).IR2(np.array(df_train_predictions['strategy'].values))
                training_IR2.append(Train_IR2)

                # Fit the model on train_X and Predicting the **VALIDATION** data
                model_val = LinearRegression().fit(train_X, train_Y)
                val_predictions = self.create_one_step_predictions(train_X, val_X, model_val)
                val_inversed = self.scaler_Y.inverse_transform(val_Y)

                df_val_predictions = pd.DataFrame(data={'Close': val_inversed.flatten(), 'Predictions': val_predictions.flatten()})
                df_val_predictions = BacktestStrategy(None, df_val_predictions, self.transaction_cost).backtesting()

                Val_IR2 = PerformanceMetrics(None, None, None, None).IR2(np.array(df_val_predictions['strategy'].values))
                validation_IR2.append(Val_IR2)

                # Fit the model and Predicting the **TEST** data
                combined_train_val_X = np.concatenate([train_X, val_X]) 
                combined_train_val_Y = np.concatenate([train_Y, val_Y]) 

                model_test = LinearRegression().fit(combined_train_val_X, combined_train_val_Y)
                test_predictions = self.create_one_step_predictions(combined_train_val_X, test_X, model_test)
                test_inversed = self.scaler_Y.inverse_transform(test_Y)

                df_test_predictions = pd.DataFrame(data={'Close': test_inversed.flatten(), 'Predictions': test_predictions.flatten()})
                df_test_predictions = BacktestStrategy(None, df_test_predictions, self.transaction_cost).backtesting()

                test_IR2 = PerformanceMetrics(None, None, None, None).IR2(np.array(df_test_predictions['strategy'].values))
                testing_IR2.append(test_IR2)

                config_all.append(cfg)
            
            df_finding_the_best_configuration = pd.DataFrame(
                data = {
                    'Config': config_all,
                    'Training IR2': training_IR2,
                    'Validation IR2': validation_IR2,
                    'Testing IR2': testing_IR2
                }
            )
            df_finding_the_best_configuration['custom_score'] = df_finding_the_best_configuration['Testing IR2']
            # df_finding_the_best_configuration['custom_score'] =  abs(df_finding_the_best_configuration['Validation IR2'] - df_finding_the_best_configuration['Training IR2'])
            
            # df_finding_the_best_configuration['custom_score'] = np.where(df_finding_the_best_configuration['Validation IR2']==0, np.nan, df_finding_the_best_configuration['custom_score'])

            # print(df_finding_the_best_configuration)
            
            # try:    
            best_model_configuration = df_finding_the_best_configuration.loc[df_finding_the_best_configuration['custom_score'].idxmax()]['Config']
            # except:
            #     best_model_configuration = df_finding_the_best_configuration.loc[df_finding_the_best_configuration['Training IR2'].idxmax()]['Config']
            
            # printing the best model configuration
            print(f'The Best Model Configuration Is: {best_model_configuration}')

            # Predict on the real test data set / Train the Training and Validation data combinedly
            train_X, train_Y, val_X, val_Y, test_X, test_Y = self.data_preperation(df, ranges, best_model_configuration['num_lags'], best_model_configuration['RSI_window'], i, best_model_configuration['RV_window'])

            combined_train_val_X = np.concatenate([train_X, val_X]) 
            combined_train_val_Y = np.concatenate([train_Y, val_Y]) 

            # Fit the model
            model = LinearRegression().fit(combined_train_val_X, combined_train_val_Y)
            
            # Predict
            history = combined_train_val_X # temporary name until i dont create more linear regressions.
            N_test_observations = len(test_X)

            model_predictions = np.zeros((0,1)).reshape(-1,1)
            for time_point in range(N_test_observations):
                y_hat = model.predict(history[-1:])
                model_predictions = np.concatenate((model_predictions, y_hat))

                real_test_value = test_X[time_point].reshape(1,-1)
                history = np.concatenate((history, real_test_value))
            # model_predictions = model.predict(test_X)
            inversed_predictions = self.scaler_Y.inverse_transform(model_predictions)
            ALL_MODEL_PREDICTIONS = np.concatenate((ALL_MODEL_PREDICTIONS, inversed_predictions))
            ACTUAL_VALUES = np.concatenate((ACTUAL_VALUES, self.scaler_Y.inverse_transform(test_Y)))

            print(f'Walk Forward Window {i} has been calculated')
        
        DF_PREDICTIONS = pd.DataFrame(
            data={
                'Inversed_Close': ACTUAL_VALUES.flatten(),
                'Close': self.get_data()[self.training_bars+self.validation_bars:]['Close'],
                'Predictions': ALL_MODEL_PREDICTIONS.flatten()
                }
                )

        DF_PREDICTIONS.to_csv('./Results/df_predictions.csv')
        self.get_data()['Close'].to_csv('./Results/df.csv')
        
        return ACTUAL_VALUES, ALL_MODEL_PREDICTIONS
    
class PerformanceMetrics:

    def __init__(self, NAZWA_1, NAZWA_2, TAB_BH, TABL_ALGO):

        self.nazwa_1 = NAZWA_1
        self.nazwa_2 = NAZWA_2

        self.tab_BH = TAB_BH
        self.tab_Algo = TABL_ALGO

    def EquityCurve_na_StopyZwrotu(self, tab):
        ret = [(tab[i + 1] / tab[i]) - 1 for i in range(len(tab) - 1)]
        return ret

    def ARC(self, tab):
        temp = self.EquityCurve_na_StopyZwrotu(tab)
        lenth = len(tab)
        a_rtn = 1
        for i in range(len(temp) - 1):
            rtn = (1 + temp[i])
            a_rtn = a_rtn * rtn
        if a_rtn <= 0:
            a_rtn = 0
        else:
            a_rtn = math.pow(a_rtn, (252 / lenth)) - 1
        return 100 * a_rtn

    def MaximumDrawdown(self, tab):
        eqr = np.array(self.EquityCurve_na_StopyZwrotu(tab))
        cum_returns = np.cumprod(1 + eqr)
        cum_max_returns = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_max_returns - cum_returns) / cum_max_returns
        max_drawdown = np.max(drawdowns)
        return max_drawdown * 100

    def ASD(self, tab):
        return ((((252) ** (1 / 2))) * np.std(self.EquityCurve_na_StopyZwrotu(tab))) * 100

    def sgn(self, x):
        if x == 0:
            return 0
        else:
            return int(abs(x) / x)

    def MLD(self, tab):
        if tab == []:
            return 1
        if tab != []:
            i = np.argmax(np.maximum.accumulate(tab) - tab)
            if i == 0:
                return len(tab) / 252.03
            j = np.argmax(tab[:i])
            MLD_end = -1
            for k in range(i, len(tab)):
                if (tab[k - 1] < tab[j]) and (tab[j] < tab[k]):
                    MLD_end = k
                    break
            if MLD_end == -1:
                MLD_end = len(tab)

        return abs(MLD_end - j) / 252.03

    def IR1(self, tab):
        aSD = self.ASD(tab)
        ret = self.ARC(tab)
        licznik = ret
        mianownik = aSD
        val = licznik / mianownik
        if mianownik == 0:
            return 0
        else:
            return max(val, 0)

    def IR2(self, tab):
        aSD = self.ASD(tab)
        ret = self.ARC(tab)
        md = self.MaximumDrawdown(tab)
        licznik = (ret ** 2) * self.sgn(ret)
        mianownik = aSD * md
        val = licznik / mianownik
        if mianownik == 0:
            return 0
        else:
            return max(val, 0)

    def wyniki(self):
        print('\n')
        print('Wyniki dla {0} prezentują się następująco: \n'.format(self.nazwa_1))
        print('ASD {0} \n MD {1}% \n ARC {2}% \n MLD {3} lat \n IR1 {4} \n IR2 {5}'.format(
            "%.2f" % self.ASD(self.tab_BH), "%.2f" % self.MaximumDrawdown(self.tab_BH), "%.2f" % self.ARC(self.tab_BH),
            "%.2f" % self.MLD(self.tab_BH), "%.4f" % self.IR1(self.tab_BH), "%.4f" % self.IR2(self.tab_BH)))
        print('\n')
        print('Wyniki dla {0} prezentują się następująco: \n'.format(self.nazwa_2))
        print('ASD {0} \n MD {1}% \n ARC {2}% \n MLD {3} lat \n IR1 {4} \n IR2 {5}'.format(
            "%.2f" % self.ASD(self.tab_Algo), "%.2f" % self.MaximumDrawdown(self.tab_Algo), "%.2f" % self.ARC(self.tab_BH),
            "%.2f" % self.MLD(self.tab_Algo), "%.4f" % self.IR1(self.tab_Algo), "%.4f" % self.IR2(self.tab_Algo)))

    def porownanie(self):
        ASD_bh = self.ASD(self.tab_BH)
        ASD_alg = self.ASD(self.tab_Algo)

        MD_bh = self.MaximumDrawdown(self.tab_BH)
        MD_alg = self.MaximumDrawdown(self.tab_Algo)

        ARC_bh = self.ARC(self.tab_BH)
        ARC_alg = self.ARC(self.tab_Algo)

        MLD_bh = self.MLD(self.tab_BH)
        MLD_alg = self.MLD(self.tab_Algo)

        IR1_bh = self.IR1(self.tab_BH)
        IR1_alg = self.IR1(self.tab_Algo)

        IR2_bh = self.IR2(self.tab_BH)
        IR2_alg = self.IR2(self.tab_Algo)
        
        print('\n')
        if ASD_bh >= ASD_alg:
            print('\033[92m Strategia lepsza od BH pod względem ASD')
        else:
            print('\033[91m Strategia gorsza od BH pod względem ASD')

        if ARC_alg >= ARC_bh:
            print('\033[92m Strategia lepsza od BH pod względem ARC')
        else:
            print('\033[91m Strategia gorsza od BH pod względem ARC')

        if MD_bh >= MD_alg:
            print('\033[92m Strategia lepsza od BH pod względem MD')
        else:
            print('\033[91m Strategia gorsza od BH pod względem MD')

        if MLD_bh >= MLD_alg:
            print('\033[92m Strategia lepsza od BH pod względem MLD')
        else:
            print('\033[91m Strategia gorsza od BH pod względem MLD')

        if IR1_alg >= IR1_bh:
            print('\033[92m Strategia lepsza od BH pod względem IR1')
        else:
            print('\033[91m Strategia gorsza od BH pod względem IR1')

        if IR2_alg >= IR2_bh:
            print('\033[92m Strategia lepsza od BH pod względem IR2')
        else:
            print('\033[91m Strategia gorsza od BH pod względem IR2')

class BacktestStrategy:
    def __init__(self, DF_INDEX, DF_PREDICTIONS, TRANSACTION_COST):

        self.df_index = DF_INDEX
        self.df_predictions = DF_PREDICTIONS

        self.transac_cost = TRANSACTION_COST

        pass

    # plot inversed predictions with closing price and inversed actual closing price
    def fig_pred_act(self):   
        fig_train_test_vs_prediction = go.Figure()

        fig_train_test_vs_prediction.add_trace(
            go.Scatter(x=self.df_index.index, y=self.df_index['Close'], name="The whole dataset"),
        )

        fig_train_test_vs_prediction.add_trace(
            go.Scatter(x=self.df_predictions.index, y=self.df_predictions['Close'], name="Test Data"),
        )

        fig_train_test_vs_prediction.add_trace(
            go.Scatter(x=self.df_predictions.index, y=self.df_predictions['Predictions'], name="Prediction")
        )

        fig_train_test_vs_prediction.update_layout(
            title={
                'text': "Display of real values vs Predictions",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title="Date",
            yaxis_title="Closing Price",
            legend_title="Data",
            template="plotly_dark"

        )

        return fig_train_test_vs_prediction.show()
    
    # calculating the transaction cost
    def transaction_cost(self, df_equity_curve, transaction_cost):

        prev_position = df_equity_curve['Position'].iloc[0]

        for i, position in enumerate(df_equity_curve['Position']):
            if position != prev_position:
                df_equity_curve['strat_return'].iloc[i] -= transaction_cost        
            prev_position = position
        
        return df_equity_curve
    
    # calculating the equity line
    def backtesting(self):

        position_LO = np.where(self.df_predictions['Predictions'].shift(-1)>self.df_predictions['Close'],1,0)
        self.df_predictions['Position'] = position_LO

        self.df_predictions['strat_return'] = self.df_predictions['Close'].pct_change().dropna()
        self.df_predictions['bnh_return'] = self.df_predictions['Close'].pct_change().dropna()

        self.df_predictions = self.transaction_cost(self.df_predictions, self.transac_cost) 

        self.df_predictions["strategy"] = (self.df_predictions["strat_return"] * self.df_predictions['Position'].shift(1))
        self.df_predictions["strategy"] = (1 + self.df_predictions["strategy"].fillna(0)).cumprod()

        self.df_predictions['buy_n_hold'] = (1 + self.df_predictions['bnh_return'].fillna(0)).cumprod()
        
        return self.df_predictions
    
    def fig_strategies(self):
        fig_equity_curve_strategy = go.Figure()

        fig_equity_curve_strategy.add_trace(
            go.Scatter(x=self.df_predictions.index, y=self.df_predictions['strategy'], name="Equity Curve Strategy"),
        )

        fig_equity_curve_strategy.add_trace(
            go.Scatter(x=self.df_predictions.index, y=self.df_predictions['buy_n_hold'], name="Benchmark"),
        )

        fig_equity_curve_strategy.update_layout(
            title={
                'text': "Strategy",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Data",
            template="plotly_dark"

        )

        return fig_equity_curve_strategy.show()
    
    def space(self):
        return "/n/n"
    
    def run_all_methods(self):

        self.df_predictions = self.backtesting()

        self.df_predictions.to_csv('./Results/df_Equity_Curve.csv')

        PM = PerformanceMetrics('Buy & Hold Strategy', 'Equity Curve Strategy', 
                                np.array(self.df_predictions['buy_n_hold'].values), np.array(self.df_predictions['strategy'].values))
        
        return self.fig_pred_act(), self.fig_strategies(), PM.wyniki(), PM.porownanie()
        



