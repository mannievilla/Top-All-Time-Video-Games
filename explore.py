# import data
import prepare
from prepare import split_data


#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# stats testing
from scipy import stats


############################# Plotting the NA Region #############################
    


def plot_na_region():

    # calling the data
    train, validate, test = split_data()

    # masking the data
    top_ranked_country = train[['Game', 'North America', 'Europe', 'Japan', 'Rest of World', 'Top Ranked']]

    # creating subset for plotting
    region_sales = top_ranked_country[top_ranked_country['Top Ranked'] == 1].sort_values(by='North America', ascending=False).drop_duplicates().head(10)
    region_sales_not_rank = top_ranked_country[top_ranked_country['Top Ranked'] == 0].sort_values(by='North America', ascending=False).head(10)

    # plotting the data
    # fig, ax = plt.subplots(figsize=(10, 8))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,8))      

    sns.barplot(data=region_sales, x='Game', y='North America', ax=ax[0])
    ax[0].set_xticklabels(region_sales['Game'], rotation=90)  # Set x-axis tick labels on the first axis

    sns.barplot(data=region_sales_not_rank, x='Game', y='North America', ax=ax[1])
    ax[1].set_xticklabels(region_sales_not_rank['Game'], rotation=90)

    # Set y-axis tick positions and labels for the second axis
    plt.sca(ax[1])
    plt.yticks(range(0, 81, 10))
    
    plt.tight_layout()
    plt.show()



############################# Plotting the Meta Scores #############################




def plot_meta():


    # calling the data
    train, validate, test = split_data()

    #plotting the data
    sns.stripplot(data= train, x='Top Ranked' , y='Meta Score', marker='o', edgecolor='black', s=3, hue='Top Ranked')
    plt.axhline(train['Meta Score'].mean(), color='red', linestyle='dashed', label='Mean')
    plt.axhline(train[train['Top Ranked']==1]['Meta Score'].mean(), color='green', linestyle='dashed', label='Mean')

    plt.legend()
    plt.show()




############################# Plotting the User Reviews #############################


def plot_user():

    # calling the data
    train, validate, test = split_data()

    #plotting the data
    sns.stripplot(data= train, x='Top Ranked' , y='User Review', marker='o', edgecolor='black', s=3, hue='Top Ranked')
    plt.axhline(train['User Review'].mean(), color='red', linestyle='dashed', label='Mean')
    plt.axhline(train[train['Top Ranked']==1]['User Review'].mean(), color='green', linestyle='dashed', label='Mean')
    plt.legend()
    plt.show()




############################# Plotting the Year #############################


    

def plot_year():


    # calling the data
    train, validate, test = split_data()

    #creating the subset
    year_one = train[train['Top Ranked']==1]

    #plotting the data
    plt.figure(figsize=(12, 6))
    sns.countplot(data=year_one, x='Year')
    plt.xticks(rotation=45)
    plt.show()




############################# Testing the Meta Score #############################



def t_test_meta():

    # calling the data
    train, validate, test = split_data()

    # stats testing for with one sample t-test
    t, p = stats.ttest_1samp(train[train['Top Ranked']==1]['Meta Score'], train['Meta Score'].mean())
    
    return t, p




############################# Testing the User Review #############################




def t_test_user():

    # calling the data
    train, validate, test = split_data()

    # stats testing for User Review with one sample t-test
    t, p = stats.ttest_1samp(train[train['Top Ranked']==1]['User Review'], train['User Review'].mean())
    return t, p