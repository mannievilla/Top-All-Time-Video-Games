import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

from bs4 import BeautifulSoup

import os


################################### Scrape IGN for Top 500 ###################################




def get_ign_500_games():

    # Save the DataFrame to a CSV file
    filename = 'ign_games.csv'


    # Check if the file exists
    if os.path.isfile(filename):
        # Read the CSV file into a DataFrame
        ign_games = pd.read_csv(filename, index_col=0)

    else:
        

        base_url = "https://www.ign.com/playlist/suerowned/lists/the-mathematically-determined-500-best-video-games-of-all-time"
        
        # Use a web driver (e.g., Chrome) to open the webpage
        driver = webdriver.Chrome()
        driver.get(base_url)
        
        # Scroll down to load additional data dynamically
        scroll_pause_time = 1  # Adjust the pause time as needed
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Get the page source after scrolling
        page_source = driver.page_source
        
        # Close the web driver
        driver.quit()
        
        soup = BeautifulSoup(page_source, 'html.parser')
        divs = soup.find_all('figure', {'class':'jsx-855987706 figure-tile object-tile small'})
        
        # Creates the list and position
        ign_games = []
        game_position = 1
        
        for div in divs:
            a_tag = div.find('figcaption')
            if a_tag is not None:
                game_title = a_tag.getText()
                ign_games.append((game_position, game_title))
            game_position += 1
        
        # Creates a dataframe
        ign_games = pd.DataFrame(ign_games)
        
        # Renames the columns
        ign_games = ign_games.rename(columns={0: 'IGN Position', 1: 'Game'})

        # removes the bracketed text in game title
        ign_games['Game'] = ign_games['Game'].str.replace(r'\[.*?\]', '', regex=True)
        
        # removes the paranthesis in game title
        ign_games['Game'] = ign_games['Game'].str.replace(r'\(.*?\)', '', regex=True)

        ign_games.reset_index()

        # Save the DataFrame to the CSV file
        ign_games.to_csv(filename)

    
    return ign_games



################################### Acquire Data ###################################


def acquire_data():

    # Creating list to loop and store the new dataframes
    csv_list = ['backloggd_games', 'PS4_GamesSales', 'Video_Games_Sales_as_at_22_Dec_2016', 'XboxOne_GameSales', 'all_games']
    df_list = []

    # Looping throgh the JSON file names
    for item in csv_list:
        try:
            # Specify the encoding when reading the CSV file
            df = pd.read_csv(item + '.csv', encoding='utf-8')
            df_list.append(df)
        except UnicodeDecodeError:
            # Try a different encoding if utf-8 doesn't work
            try:
                df = pd.read_csv(item + '.csv', encoding='latin-1')
                df_list.append(df)
            except UnicodeDecodeError:
                # If still unsuccessful, try other encodings
                # Add more encodings to try as needed
                df = pd.read_csv(item + '.csv', encoding='ISO-8859-1')
                df_list.append(df)

        
    return df_list



################################### Clean the Data ###################################



def clean_data():

    allgen, ps, nin, xbox, meta = acquire_data()

    #### PLAYSTATION ####

    # removing any rows with no sales on record
    ps = ps[ps[['North America', 'Europe', 'Japan', 'Rest of World', 'Global']].sum(axis=1)>0]

    #### NINTENEDO ####

    # Dropping columns
    nin.drop(columns=['Platform', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating'], inplace=True)

    # rename these columns
    nin = nin.rename(columns={'Name': 'Game', 'Year_of_Release': 'Year', 'NA_Sales': 'North America', 'EU_Sales': 'Europe', 
                          'JP_Sales': 'Japan', 'Other_Sales': 'Rest of World', 
                          'Global_Sales': 'Global'})
    
    #### XBOX ####

    # dropping columns
    xbox = xbox[xbox[['North America', 'Europe', 'Japan', 'Rest of World', 'Global']].sum(axis=1)>0]


    #### METACRITIC ####

    # Dropping unnecessary columns
    meta.drop(columns=['platform', 'release_date', 'summary'], inplace=True)

    # rename col
    meta.rename(columns={'name': 'Game', 'meta_score': 'Meta Score', 'user_review': 'User Review'}, inplace=True)

    # remove all 'tbd' values from User Review
    meta = meta[meta['User Review']!='tbd']

    # Converting to integer
    meta['User Review'] = pd.to_numeric(meta['User Review'], errors='coerce')

    # Getting the averages of all the scores from each itteration of the game
    meta = meta.groupby('Game').mean()

    # reseting the index
    meta = meta.reset_index()

    return ps, nin, xbox, meta




################################### Join the Data ###################################




def join_data():



    # calling the df
    ps, nin, xbox, meta = clean_data()
    ign_games = get_ign_500_games()

    # merging xbox and ps df
    xb_ps = pd.merge(xbox, ps, on='Game', how='outer')


    # Filling in certain null values to be able to complete an operation
    columns_to_fill = ['North America_x',
                        'Europe_x', 'Japan_x', 'Rest of World_x', 'Global_x',
                        'North America_y', 'Europe_y', 'Japan_y',
                        'Rest of World_y', 'Global_y']
    xb_ps[columns_to_fill] = xb_ps[columns_to_fill].fillna(0)


    # Summing the two seperate categories
    xb_ps['North America'] = xb_ps['North America_x'] + xb_ps['North America_y']
    xb_ps['Europe'] = xb_ps['Europe_x'] + xb_ps['Europe_y']
    xb_ps['Japan'] = xb_ps['Japan_x'] + xb_ps['Japan_y']
    xb_ps['Rest of World'] = xb_ps['Rest of World_x'] + xb_ps['Rest of World_y']
    xb_ps['Global'] = xb_ps['Global_x'] + xb_ps['Global_y']


    # Dropping redundant columns
    xb_ps.drop(columns=['North America_x',
        'Europe_x', 'Japan_x', 'Rest of World_x', 'Global_x',
        'North America_y', 'Europe_y', 'Japan_y',
        'Rest of World_y', 'Global_y',], inplace=True)
    
    # Looping through the values in column_x and comapring to column_y and replacing or leaving the value unless it is null, It will return the the value in column_y
    xb_ps['Year'] = xb_ps.apply(lambda row: row['Year_x'] if pd.isnull(row['Year_y']) else row['Year_y'], axis=1)

    # Looping through the values in column_x and comapring to column_y and replacing or leaving the value unless it is null, It will return the the value in column_y
    xb_ps['Genre'] = xb_ps.apply(lambda row: row['Genre_x'] if pd.isnull(row['Genre_y']) else row['Genre_y'], axis=1)

    # Looping through the values in column_x and comapring to column_y and replacing or leaving the value unless it is null, It will return the the value in column_y
    xb_ps['Publisher'] = xb_ps.apply(lambda row: row['Publisher_x'] if pd.isnull(row['Publisher_y']) else row['Publisher_y'], axis=1)

    # dropping these columns
    xb_ps.drop(columns=['Year_x', 'Year_y', 'Genre_x', 'Genre_y', 'Publisher_x', 'Publisher_y'], inplace=True)

    # Merginf nin df to xb_ps and creating a total_sales df
    total_sales = pd.merge(xb_ps, nin, on='Game', how='outer')

    # Filling in certain null values to be able to complete an operation
    columns_to_fill = ['North America_x',
                        'Europe_x', 'Japan_x', 'Rest of World_x', 'Global_x',
                        'North America_y', 'Europe_y', 'Japan_y',
                        'Rest of World_y', 'Global_y']
    total_sales[columns_to_fill] = total_sales[columns_to_fill].fillna(0)

    # Summing the two seperate categories
    total_sales['North America'] = total_sales['North America_x'] + total_sales['North America_y']
    total_sales['Europe'] = total_sales['Europe_x'] + total_sales['Europe_y']
    total_sales['Japan'] = total_sales['Japan_x'] + total_sales['Japan_y']
    total_sales['Rest of World'] = total_sales['Rest of World_x'] + total_sales['Rest of World_y']
    total_sales['Global'] = total_sales['Global_x'] + total_sales['Global_y']

    # Looping throught the values in column_x and comapring to column_y and replacing or leaving the value unless it is null, It will return the the value in column_y
    total_sales['Year'] = total_sales.apply(lambda row: row['Year_x'] if pd.isnull(row['Year_y']) else row['Year_y'], axis=1)

    # Looping throught the values in column_x and comapring to column_y and replacing or leaving the value unless it is null, It will return the the value in column_y
    total_sales['Genre'] = total_sales.apply(lambda row: row['Genre_x'] if pd.isnull(row['Genre_y']) else row['Genre_y'], axis=1)

    # Looping throught the values in column_x and comapring to column_y and replacing or leaving the value unless it is null, It will return the the value in column_y
    total_sales['Publisher'] = total_sales.apply(lambda row: row['Publisher_y'] if pd.isnull(row['Publisher_x']) else row['Publisher_x'], axis=1)

    # Dropping Columns
    total_sales.drop(columns=['North America_x',
        'Europe_x', 'Japan_x', 'Rest of World_x', 'Global_x',
        'North America_y', 'Europe_y', 'Japan_y',
        'Rest of World_y', 'Global_y', 'Year_x', 'Year_y',
        'Genre_x', 'Genre_y', 'Publisher_x', 'Publisher_y', 'Pos'], inplace=True)
    

    # creating a new dataframe to aggregate the columns with the sum but only the features chosen not Year
    total_grp_sales = total_sales.groupby('Game').agg({'North America': 'sum', 'Europe': 'sum', 'Japan': 'sum', 
                                                        'Rest of World': 'sum', 'Global': 'sum'
                                                        })
    total_grp_sales = total_grp_sales.reset_index()

    # Creating a Year df that to add back to the total_grp_sales
    year = pd.DataFrame(total_sales[['Game', 'Year']])

    # Merging the year and the total_group_sales df
    total_grp_sales = pd.merge(total_grp_sales, year, on='Game', how='inner')

    # Dropping duplicaate values
    total_grp_sales = total_grp_sales.drop_duplicates()

    # Dropping rows with null values
    total_grp_sales = total_grp_sales.dropna()

    # Creating the final version of the df that will be used...... FINALLY!
    df = pd.merge(total_grp_sales, meta, on='Game', how='inner')    

    # Creating the final version of the df that will be used...... FINALLY! For real..
    df = pd.merge(df, ign_games, on='Game', how='left')  

    return df, total_sales



################################### Clean the Joined Data ###################################




def clean_join_data():

    df, total_sales = join_data()

    # Round all the metascores to ones
    df['Meta Score'] = df['Meta Score'].apply(lambda x: int(round(x)))

    # Rounding to two decimal places
    df['User Review'] = round(df['User Review'],2)

    # Creating a genre df to add to the final df
    genre = total_sales[['Game', 'Genre']] 

    # Merging the genres to the final df
    df = pd.merge(df, genre, on='Game', how='left')

    # dropping duplicates
    df = df.drop_duplicates()

    # reseting and dropping old index
    df.reset_index(drop=True, inplace= True)

    # Dropping the Genree with values less than 200 to create meaningful categories
    genre_counts = df['Genre'].value_counts()
    selected_genres = genre_counts[genre_counts > 200].index
    df = df[df['Genre'].isin(selected_genres)]

    return df


