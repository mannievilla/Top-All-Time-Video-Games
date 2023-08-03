import pandas as pd
from acquire import clean_join_data

#import sklearn modules
from sklearn.model_selection import train_test_split


################################### Prep the Data ###################################

def prepped_data():

    # calling the data
    df = clean_join_data()

    # replacing integer values in IGN Position with Yes
    df['IGN Position'].where(df['IGN Position'].isna(), "Yes", inplace=True)

    # replacing na values in IGN Position with No
    df["IGN Position"].fillna("No", inplace = True)


    return df


################################### Split the Data ###################################

def split_data():

    # calling the data
    df = prepped_data()

    df_dummies = pd.get_dummies(df['Genre'], drop_first=False)
    # Concatenate the original DataFrame and the dummy variables DataFrame
    df = pd.concat([df, df_dummies], axis=1)


    df_dummies = pd.get_dummies(df['IGN Position'], drop_first=True)
    # Concatenate the original DataFrame and the dummy variables DataFrame
    df = pd.concat([df, df_dummies], axis=1)


    # dropping the original feature
    df.drop(columns='Genre', inplace=True, axis=1)


    # dropping the original feature
    df.drop(columns='IGN Position', inplace=True, axis=1)


    # renaming the new dummy column
    df.rename(columns={'Yes': 'Top Ranked'}, inplace=True)


    # split test off, 20% of original df size. 
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=42)

    # split validate off, 30% of what remains (24% of original df size)
    # thus train will be 56% of original df size. 
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=42)
    
    return train, validate, test



################################### X y Split ###################################




def X_y_split():
    
    train, validate, test = split_data()

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['Top Ranked'])
    y_train = train['Top Ranked']
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['Top Ranked'])
    y_validate = validate['Top Ranked']
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['Top Ranked'])
    y_test = test['Top Ranked']
    
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test