import pandas as pd
"""
Custom feature engineering functions 

"""
def add_total_charges(df: pd.DataFrame):
   # df['Total calls'] = df.loc[:,['Total day calls','Total eve calls','Total night calls','Total intl calls']].sum(axis=1)
    df['Total charges'] = df.loc[:,['Total day charge','Total eve charge','Total night charge','Total intl charge']].sum(axis=1)
    return df


def group_state(df):

    '''
    states: States ordered in ascending order with high churn customers
    
    '''
    states = ['TX', 'MD', 'NJ', 'MN', 'NV', 'MI', 'NY', 'MS', 'AR', 'ME', 'SC', 'CT',
       'MT', 'WA', 'OH', 'KS', 'NC', 'NH', 'GA', 'UT', 'WY', 'PA', 'MA', 'DE',
       'OR', 'WV', 'OK', 'AL', 'FL', 'CO', 'IN', 'SD', 'KY', 'VT', 'CA', 'TN',
       'MO', 'DC', 'ID', 'NM', 'IL', 'ND', 'NE', 'WI', 'VA', 'AK', 'AZ', 'IA',
       'LA', 'RI', 'HI']
    
    df['State group churn'] = 0
    
    filt = df['State'].isin(states[:17])
    df.loc[filt,'State group churn'] = 3
    
    filt = df['State'].isin(states[17:34])
    df.loc[filt,'State group churn'] = 2
    
    filt = df['State'].isin(states[34:])
    df.loc[filt,'State group churn'] = 1
    
    filt = ~df['State'].isin(states)
    df.loc[filt,'State group churn'] =0

    df.drop(columns=['State'],inplace=True)

    return df


def drop_multicollinear_features(df):
    
    return df.drop(columns=['Total day charge','Total day minutes',
                            'Total eve charge','Total eve minutes',
                            'Total intl charge','Total night charge'])


