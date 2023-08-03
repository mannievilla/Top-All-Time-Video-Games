# Top All Time Video Games
 
# Project Description
 
With having so many options to play on your home console, making a decision can be hard. Don't waste time googling the best options. Let's take some concrete data and take the guessing out of the equation. Is your game a Top Ranked game, according to IGN, Lets put it throuhg the paces and through this classification model.
 
# Project Goal
 
* Discover drivers relate to being on IGN's Top 500 Games of All Time
* Use drivers to develop a machine learning model to classify the video game whether or not the video game is on the list. 
* Top Ranked is defined as being on IGN's Top 500 Games of All Time. 
 
# Initial Thoughts
 
My initial hypothesis is that drivers of Ranked Games will be Norht American Sales and Meta Scores.
 
# The Plan
 
* Aquire data Kaggle website:
    + https://www.kaggle.com/datasets/sidtwr/videogames-sales-dataset
    + https://www.kaggle.com/datasets/deepcontractor/top-video-games-19952021-metacritic
    + https://www.ign.com/playlist/suerowned/lists/the-mathematically-determined-500-best-video-games-of-all-time
 
* Prepare data
    * Manually downloaded the video game data from Kaggle
    * Web Scraped te data from IGN Top 500 List
    * Cleaned up the names in all datasets to able to use that key to join all names
    * Joined all datasets
    * Obtained average sales after combinung games based on what system it was sold on
    * Summed the sales after combining games based on what system it was sold on
    * Removed null values while joining datasets one at a time
    * Removed any duplicated rows after joining datasets
    * Engineered features with dummy columns
        + Genre
        + Top Ranked

 
* Explore data in search of drivers for churn
   * Answer the following initial questions
       + How do regional sales relate to Ranking in Top 500?
       + How does meta score relate to Top 500 Rank?
       + How does User Review Score relate to Top 500 Rank?
       + How do Years relate to Top Ranked?
      
* Develop a Model to predict if a customer will churn
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

Alongside the fields: Name, Platform, Year_of_Release, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales, we have:-
| Feature | Definition |
|:--------|:-----------|
|Game| Name of the video game|
|North America| Sales in the region in million dollars|
|Europe| Sales in the region in million dollars|
|Japan| Sales in the region in million dollars|
|Rest of World| Sales in the region in million dollars|
|Global| Sales in the region in million dollars|
|Meta Score| Average score of games from Metacritic|
|Year| Year game was released|
|User Review| Average score from users on Metacritic|
|IGN Position|Yes or No, Web Scraped List from IGN's Top 500 Games of All Time|
|Engineered Features| 0 or 1, Engineered from our target, 'Top Ranked' and from 'Genre'

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from various websites and web scraping
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* Region sales or popularity do not mean a Top Ranked video game. 
* Most Top Ranked games have a Meta score of above 85
* The highest ranked games are on the Ranked list
* User scores are on average lower than the meta score. 
* The lowest scoring Ranked game has about a 3.
* A steady increase in the number of games selected starting from the 80s.
* Some games really stand the test of time
* The final model performed 3 points better than baseline (.93). There was not a whole lot of room for improvement. 
* Model improved 42% of the possible improvement room of 7 points.
* 3% is 144 possible selections accurately predicted.


 
# Recommendations
* I don't reccomend to move forward just yet. More options need to be explored. Looking into Recall may be a better metric to make sure we are properly predicting the games on the all-time list.


# Next Steps
* Find more data to the video games. Example: Rating, Price at Launch, Availability(New), Online Connectivity, Multi/Single Player. 
* Pick a model based on Recall. Accuracy is a good ocerall measure but Recall would let us know if the prediction is worth while. 
* Consider combining another webistes All Time List that could introduce other games that may have been left out.