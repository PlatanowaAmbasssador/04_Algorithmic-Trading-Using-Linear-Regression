import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# performance metrics

def EquityCurve_na_StopyZwrotu(tab):

    ret=[]

    for i in range(0,len(tab)-1):
        ret.append((tab[i+1]/tab[i])-1)

    return ret

def AbsoluteRateOfReturnFromEquityCurve(tab):
    returns = EquityCurve_na_StopyZwrotu(tab)
    if len(returns) == 0:
        return 0

    absolute_return = 1.0  # Initialize with 100% (1.0)
    
    for return_value in returns:
        absolute_return *= (1 + return_value)

    return (absolute_return - 1.0) * 100 

def ARC(tab):
    temp=EquityCurve_na_StopyZwrotu(tab)
    lenth = len(tab)
    a_rtn=1
    for i in range(len(temp)-1):
        rtn=(1+temp[i])
        a_rtn=a_rtn*rtn
    if a_rtn <= 0:
        a_rtn = 0
    else:
        a_rtn = math.pow(a_rtn,(252/lenth)) - 1
    return 100*a_rtn


def MaximumDrawdown(tab):

    eqr = np.array(EquityCurve_na_StopyZwrotu(tab))
    
    cum_returns = np.cumprod(1 + eqr)
    cum_max_returns = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_max_returns - cum_returns) / cum_max_returns
    max_drawdown = np.max(drawdowns)
    return max_drawdown*100


def ASD(tab):
    return ((((252)**(1/2)))*np.std(EquityCurve_na_StopyZwrotu(tab)))*100

def sgn(x):
    if x==0:
        return 0
    else:
        return int(abs(x)/x)

def MLD(temp):
    if temp==[]:
        return 1
    if temp !=[]:
        i=np.argmax(np.maximum.accumulate(temp)-temp)
        if i==0:
            return len(temp)/252.03
        j=np.argmax(temp[:i]) 
        MLD_end=-1
        for k in range(i,len(temp)):
            if (temp[k-1] < temp[j]) and (temp[j]<temp[k]):
                MLD_end=k
                break
        if MLD_end == -1:
            MLD_end=len(temp)

    return abs(MLD_end - j)/252.03

def IR1(tab):

    aSD=ASD(tab)

    ret=ARC(tab)#/100

    licznik=ret

    mianownik=aSD

    val=licznik/mianownik

    if mianownik==0:
        return 0
    else: 
        return max(val,0)

def IR2(tab):

    aSD=ASD(tab)

    ret=ARC(tab)#/100

    md=MaximumDrawdown(tab)#/100

    licznik=(ret**2)*sgn(ret)

    mianownik=aSD*md

    val=licznik/mianownik

    if mianownik==0:
        return 0
    else: 
        return max(val,0)

def wyniki(tab, nazwa):
    print('Wyniki dla {0} prezentują się następująco: \n'.format(nazwa))
    print('ASD {0} \n MD {1}% \n ARC {2}% \n MLD {3} lat \n IR1 {4} \n IR2 {5}'.format("%.2f" % ASD(tab),"%.2f" %  MaximumDrawdown(tab),"%.2f" %  ARC(tab),"%.2f" %  MLD(tab),"%.4f" % IR1(tab),"%.4f" % IR2(tab))) 

def porownanie(tab_Algo, tab_BH):

    ASD_bh = ASD(tab_BH)
    ASD_alg = ASD(tab_Algo)

    MD_bh = MaximumDrawdown(tab_BH)
    MD_alg = MaximumDrawdown(tab_Algo)

    ARC_bh = ARC(tab_BH)
    ARC_alg = ARC(tab_Algo)

    MLD_bh = MLD(tab_BH)
    MLD_alg = MLD(tab_Algo)

    IR1_bh = IR1(tab_BH)
    IR1_alg = IR1(tab_Algo)

    IR2_bh = IR2(tab_BH)
    IR2_alg = IR2(tab_Algo)

    if ASD_bh>=ASD_alg:
        print('Strategia lepsza od BH pod względem ASD')
    else:
        print('\033[91m Strategia gorsza od BH pod względem ASD')

    if ARC_alg>=ARC_bh:
        print('\033[92m Strategia lepsza od BH pod względem ARC')
    else:
        print('\033[91m Strategia gorsza od BH pod względem ARC')

    if MD_bh>=MD_alg:
        print('\033[92m Strategia lepsza od BH pod względem MD')
    else:
        print('\033[91m Strategia gorsza od BH pod względem MD')

    if MLD_bh>=MLD_alg:
        print('\033[92m Strategia lepsza od BH pod względem MLD')
    else:
        print('\033[91m Strategia gorsza od BH pod względem MLD')

    if IR1_alg>=IR1_bh:
        print('\033[92m Strategia lepsza od BH pod względem IR1')
    else:
        print('\033[91m Strategia gorsza od BH pod względem IR1')

    if IR2_alg>=IR2_bh:
        print('\033[92m Strategia lepsza od BH pod względem IR2')
    else:
        print('\033[91m Strategia gorsza od BH pod względem IR2')

def add_asterisks(column):
    if column.name == '':
        return column
    if column.name in ['ARC(%)', 'IR*(%)', 'IR**(%)']:
        max_value = column.replace('', np.nan).max()  # Replace empty strings with NaN for max calculation
        column = column.apply(lambda x: f"**{x}**" if x == max_value and x == max_value else x)
    elif column.name in ['ASD(%)', 'MD(%)', 'MLD']:
        # Convert non-empty strings to float and replace empty strings with NaN for min calculation
        # column = column.apply(lambda x: float(x) if x != '' else np.nan)
        min_value = max_value = column.replace('', np.nan).min()
        column = column.apply(lambda x: f"**{x}**" if x == min_value and not np.isnan(x) else x)
    return column

def performance_metrics(STRING, tab_benchmark, tab_arima, STOCK_NAME):
    try:
        tab_benchmark = np.array(tab_benchmark['buy_n_hold'].values)
    except TypeError:
        tab_benchmark = None
    
    try:
        tab_arima = np.array(tab_arima['strategy'].values)
    except TypeError:
        tab_arima = None
    

    df_data = {
        "": ["", STOCK_NAME, "Linear Regression",],
        "ARC(%)": [],
        "ASD(%)": [],
        "MD(%)": [],
        "MLD": [],
        "IR*(%)": [],
        "IR**(%)": [],
    }

    if isinstance(tab_arima, str):
        for _ in range(6):
            df_data["ARC(%)"].append(None)
            df_data["ASD(%)"].append(None)
            df_data["MD(%)"].append(None)
            df_data["MLD"].append(None)
            df_data["IR*(%)"].append(None)
            df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append("")
        df_data["ASD(%)"].append("")
        df_data["MD(%)"].append("")
        df_data["MLD"].append("")
        df_data["IR*(%)"].append("")
        df_data["IR**(%)"].append("")

        if isinstance(tab_benchmark, str) or tab_benchmark is None:
            df_data["ARC(%)"].append(None)
            df_data["ASD(%)"].append(None)
            df_data["MD(%)"].append(None)
            df_data["MLD"].append(None)
            df_data["IR*(%)"].append(None)
            df_data["IR**(%)"].append(None)
        else:
            df_data["ARC(%)"].append(round(ARC(tab_benchmark), 2))
            df_data["ASD(%)"].append(round(ASD(tab_benchmark), 2))
            df_data["MD(%)"].append(round(MaximumDrawdown(tab_benchmark), 2))
            df_data["MLD"].append(round(MLD(tab_benchmark), 2))
            df_data["IR*(%)"].append(round(IR1(tab_benchmark)*100, 2))
            df_data["IR**(%)"].append(round(IR2(tab_benchmark)*100, 2))
        
        if isinstance(tab_arima, str) or tab_arima is None:
            df_data["ARC(%)"].append(None)
            df_data["ASD(%)"].append(None)
            df_data["MD(%)"].append(None)
            df_data["MLD"].append(None)
            df_data["IR*(%)"].append(None)
            df_data["IR**(%)"].append(None)
        else:
            df_data["ARC(%)"].append(round(ARC(tab_arima), 2))
            df_data["ASD(%)"].append(round(ASD(tab_arima), 2))
            df_data["MD(%)"].append(round(MaximumDrawdown(tab_arima), 2))
            df_data["MLD"].append(round(MLD(tab_arima), 2))
            df_data["IR*(%)"].append(round(IR1(tab_arima)*100, 2))
            df_data["IR**(%)"].append(round(IR2(tab_arima)*100, 2))



    df_perf_metr = pd.DataFrame(data=df_data, index=[STRING, "",""])
    df_perf_metr = df_perf_metr.apply(add_asterisks)
    return df_perf_metr

    try:
        tab_benchmark = np.array(tab_benchmark['buy_n_hold'].values)
    except TypeError:
        tab_benchmark = None
    
    try:
        tab_LSTM = np.array(tab_LSTM['strategy'].values)
    except TypeError:
        tab_LSTM = None
    
    try:
        tab_LSTM_SENS_B = np.array(tab_LSTM_SENS_B['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_B = None
    
    try:
        tab_LSTM_SENS_C = np.array(tab_LSTM_SENS_C['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_C = None

    df_data = {
        "": [STOCK_NAME, "&nbsp;", "Base Case", "Order Range = {0-3,1,0-3}","Information Criterion = BIC", "&nbsp;",],
        "ARC(%)": [],
        "ASD(%)": [],
        "MD(%)": [],
        "MLD": [],
        "IR*(%)": [],
        "IR**(%)": [],
    }

    if isinstance(tab_benchmark, str) or tab_benchmark is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_benchmark), 2))
        df_data["ASD(%)"].append(round(ASD(tab_benchmark), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_benchmark), 2))
        df_data["MLD"].append(round(MLD(tab_benchmark), 2))
        df_data["IR*(%)"].append(round(IR1(tab_benchmark)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_benchmark)*100, 2))

    df_data["ARC(%)"].append(None)
    df_data["ASD(%)"].append(None)
    df_data["MD(%)"].append(None)
    df_data["MLD"].append(None)
    df_data["IR*(%)"].append(None)
    df_data["IR**(%)"].append(None)

    if isinstance(tab_LSTM, str) or tab_LSTM is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM)*100, 2))

    if isinstance(tab_LSTM_SENS_B, str) or tab_LSTM_SENS_B is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_B), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_B), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_B), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_B), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_B)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_B)*100, 2))

    if isinstance(tab_LSTM_SENS_C, str) or tab_LSTM_SENS_C is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_C), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_C), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_C), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_C), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_C)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_C)*100, 2))

    df_data["ARC(%)"].append(None)
    df_data["ASD(%)"].append(None)
    df_data["MD(%)"].append(None)
    df_data["MLD"].append(None)
    df_data["IR*(%)"].append(None)
    df_data["IR**(%)"].append(None)

    df_perf_metr = pd.DataFrame(data=df_data, index=["", "", STRING,"","",""])
    df_perf_metr = df_perf_metr.apply(add_asterisks_SENS)
    return df_perf_metr

def add_asterisks_SENS(column):
    if column.name is None:
        return column  # Ignore the first row

    if column.name.startswith("**Panel"):
        return column  # Keep panel labels as they are
    
    section_indices = []
    section_start = None

    # Find section boundaries based on empty rows
    for idx, value in enumerate(column):
        if pd.isna(value) or value == "":
            if section_start is not None:
                section_indices.append((section_start, idx))
                section_start = None
        elif section_start is None:
            section_start = idx

    if section_start is not None:
        section_indices.append((section_start, len(column)))

    for start, end in section_indices:
        section = column.iloc[start:end]

        if column.name in ["ARC(%)", "IR*(%)", "IR**(%)"]:
            max_value = section.max()
            if not pd.isna(max_value):
                section = section.apply(lambda x: f"**{x}**" if x == max_value and start != 0 else x)

        if column.name in ["MLD", "MD(%)", "ASD(%)"]:
            min_value = section.min()
            if not pd.isna(min_value):
                section = section.apply(lambda x: f"**{x}**" if x == min_value and start != 0 else x)

        column.iloc[start:end] = section

    return column

def performance_metrics_SENS(STRING, tab_benchmark, tab_LSTM, tab_LSTM_SENS_1, tab_LSTM_SENS_1A, tab_LSTM_SENS_2, tab_LSTM_SENS_2A, tab_LSTM_SENS_3, tab_LSTM_SENS_3A, tab_LSTM_SENS_4, STOCK_NAME, PANEL_1, PANEL_2):
    try:
        tab_benchmark = np.array(tab_benchmark['buy_n_hold'].values)
    except TypeError:
        tab_benchmark = None
    
    try:
        tab_LSTM = np.array(tab_LSTM['strategy'].values)
    except TypeError:
        tab_LSTM = None
    
    try:
        tab_LSTM_SENS_1 = np.array(tab_LSTM_SENS_1['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_1 = None
    
    try:
        tab_LSTM_SENS_1A = np.array(tab_LSTM_SENS_1A['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_1A = None

    try:
        tab_LSTM_SENS_2 = np.array(tab_LSTM_SENS_2['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_2 = None

    try:
        tab_LSTM_SENS_2A = np.array(tab_LSTM_SENS_2A['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_2A = None

    try:
        tab_LSTM_SENS_3 = np.array(tab_LSTM_SENS_3['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_3 = None
    
    try:
        tab_LSTM_SENS_3A = np.array(tab_LSTM_SENS_3A['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_3A = None

    try:
        tab_LSTM_SENS_4 = np.array(tab_LSTM_SENS_4['strategy'].values)
    except TypeError:
        tab_LSTM_SENS_4 = None

    df_data = {
        "": [STOCK_NAME, "&nbsp;", "Base Case", "Transaction Cost = 0.01%", "Transaction Cost = 0.1%", "&nbsp;",
                                    "Base Case", "Number of Lags = [1,3,5]", "Number of Lags = [24,26,28]", "&nbsp;",
                                    "Base Case", "RSI Window Length = [2,4,6]", "RSI Window Length = [25,27,29]", "&nbsp;",
                                    "Base Case", "RV Window Length = [2,4,5]", "&nbsp;",],
        "ARC(%)": [],
        "ASD(%)": [],
        "MD(%)": [],
        "MLD": [],
        "IR*(%)": [],
        "IR**(%)": [],
    }

    if isinstance(tab_benchmark, str) or tab_benchmark is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_benchmark), 2))
        df_data["ASD(%)"].append(round(ASD(tab_benchmark), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_benchmark), 2))
        df_data["MLD"].append(round(MLD(tab_benchmark), 2))
        df_data["IR*(%)"].append(round(IR1(tab_benchmark)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_benchmark)*100, 2))

    df_data["ARC(%)"].append(None)
    df_data["ASD(%)"].append(None)
    df_data["MD(%)"].append(None)
    df_data["MLD"].append(None)
    df_data["IR*(%)"].append(None)
    df_data["IR**(%)"].append(None)

    if isinstance(tab_LSTM, str) or tab_LSTM is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM)*100, 2))

    if isinstance(tab_LSTM_SENS_1, str) or tab_LSTM_SENS_1 is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_1), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_1), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_1), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_1), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_1)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_1)*100, 2))

    if isinstance(tab_LSTM_SENS_1A, str) or tab_LSTM_SENS_1A is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_1A), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_1A), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_1A), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_1A), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_1A)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_1A)*100, 2))
        
    df_data["ARC(%)"].append(None)
    df_data["ASD(%)"].append(None)
    df_data["MD(%)"].append(None)
    df_data["MLD"].append(None)
    df_data["IR*(%)"].append(None)
    df_data["IR**(%)"].append(None)

    if isinstance(tab_LSTM, str) or tab_LSTM is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM)*100, 2))

    if isinstance(tab_LSTM_SENS_2, str) or tab_LSTM_SENS_2 is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_2), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_2), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_2), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_2), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_2)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_2)*100, 2))

    if isinstance(tab_LSTM_SENS_2A, str) or tab_LSTM_SENS_2A is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_2A), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_2A), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_2A), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_2A), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_2A)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_2A)*100, 2))

    df_data["ARC(%)"].append(None)
    df_data["ASD(%)"].append(None)
    df_data["MD(%)"].append(None)
    df_data["MLD"].append(None)
    df_data["IR*(%)"].append(None)
    df_data["IR**(%)"].append(None)

    if isinstance(tab_LSTM, str) or tab_LSTM is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM)*100, 2))

    if isinstance(tab_LSTM_SENS_3, str) or tab_LSTM_SENS_3 is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_3), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_3), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_3), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_3), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_3)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_3)*100, 2))

    if isinstance(tab_LSTM_SENS_3A, str) or tab_LSTM_SENS_3A is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_3A), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_3A), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_3A), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_3A), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_3A)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_3A)*100, 2))

    df_data["ARC(%)"].append(None)
    df_data["ASD(%)"].append(None)
    df_data["MD(%)"].append(None)
    df_data["MLD"].append(None)
    df_data["IR*(%)"].append(None)
    df_data["IR**(%)"].append(None)


    if isinstance(tab_LSTM, str) or tab_LSTM is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM)*100, 2))

    if isinstance(tab_LSTM_SENS_4, str) or tab_LSTM_SENS_4 is None:
        df_data["ARC(%)"].append(None)
        df_data["ASD(%)"].append(None)
        df_data["MD(%)"].append(None)
        df_data["MLD"].append(None)
        df_data["IR*(%)"].append(None)
        df_data["IR**(%)"].append(None)
    else:
        df_data["ARC(%)"].append(round(ARC(tab_LSTM_SENS_4), 2))
        df_data["ASD(%)"].append(round(ASD(tab_LSTM_SENS_4), 2))
        df_data["MD(%)"].append(round(MaximumDrawdown(tab_LSTM_SENS_4), 2))
        df_data["MLD"].append(round(MLD(tab_LSTM_SENS_4), 2))
        df_data["IR*(%)"].append(round(IR1(tab_LSTM_SENS_4)*100, 2))
        df_data["IR**(%)"].append(round(IR2(tab_LSTM_SENS_4)*100, 2))

    df_data["ARC(%)"].append(None)
    df_data["ASD(%)"].append(None)
    df_data["MD(%)"].append(None)
    df_data["MLD"].append(None)
    df_data["IR*(%)"].append(None)
    df_data["IR**(%)"].append(None)

    df_perf_metr = pd.DataFrame(data=df_data, index=["&nbsp;", "&nbsp;", STRING,   
                                                            f"**{PANEL_1}: Transaction Cose**", "", None,
                                                            F"**{PANEL_2}: Number of Lags**", "", "", None,
                                                            F"**{PANEL_2}: RSI Window Length**", "", "", None,
                                                            F"**{PANEL_2}: RV Window Length**", "", None])
    df_perf_metr = df_perf_metr.apply(add_asterisks_SENS)
    return df_perf_metr