
# King County Housing Analysis

![King County House](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/house_final.jpg)

## Business Case

The King County region, located in the US state of Washington, is home to 2.25 million people.  An foreign investor has just moved to this area and hired our data science firm. The investor is working closely with a startup construction company that is looking to build houses in this region.  The investor would like to understand the following before making any decisions:

* What are the general trends in the housing market?
* What factors drive up the prices of houses?
* Does location have an major effect on prices?
* Is there a way to predict the prices of house for future investing purposes?

## Objective of the project
*To help the investor figure out the best attributes of a house that yields the highest profit. The concept of linear regression will be utilized to predict future prices of houses based on selected predictor variables*

#### Technologies Used:
* Pandas for Data Cleaning
* Matplotlib and Seaborn for Data Visualization
* Numpy for Basic Calculations
* SciPy/Statsmodels/Scikit Learn for Linear Regression

### Process Overview

![Process](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/process.PNG)

## Data Cleaning

* Cast columns to the appropriate data types
   * Convert data types of 'object' to appropriate numerical data type
   ![Data Type](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/Capture_3.PNG)
   
* Identify and deal with null values appropriately
  * Identify the percentage of NA values within each variable by using .value_counts(normalize = True) and then drop/replace the NA values
  ![NA values](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/Capture_4.PNG)
  
* Check for outliers and drop them from the data set
  * Use scatterplots to graph the relationship between price and each individual predictor variable. Scatterplots are a great way to observe any outliers
  * Determined that the following variables are categorical:
   * bedrooms
   * bathrooms
   * price
   * sqft_living
   * sqft_lot
   * grade
   * sqft_above
   * sqft_lot15
   
   ![Outliers](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/scatterplot_outliers.png)
   
* Filter data set based on any further outlier ranges
  * From the graphs shown above, we can reasonably say that there are not many striking outliers. However, there are data points present in some of the variables that can be eliminated to make the overall distribution a bit more representative of what we are looking for. I took into consideration the min and max values and made an educated decision on what to do with values that were below the 25th percentile and higher than the 75th percentile. Using my best judgement, I have decided that the following variables that could be filtered for are:
  ![filter](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/Capture_6.PNG)
  
* Deal with categorical variables using dummy variables
  * Created dummy variables, which is the idea of converting each category into a new column, and assign a 1 or 0 to the column
  * Applied this concept only to nominal variables, because they have no order associated with their categories
  ![categorical](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/Capture_7.PNG)
  
  ## Exploratory Data Analysis
  
  ### Question 1: Are housing prices dependent on the location? And if so, do older or newer built houses cost more?
  
  ![EDA 1](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/EDA_1.png)
  * Higher priced houses tend to be clustered in the North as opposed to the South
  * Filter for houses before 1970 and after to check for any trends
  * There are not many expensive houses before 1970 that were in the Northern region
  * Since the last half century, it seems that the more expensive houses generally tend to be cluster farther up North 
  
  ### Question 2: What is the distribution of the sizes of houses? Does having a bigger house equate to having a higher grade?

  ![EDA_2](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/EDA_2.png)
  
  * Most houses have around the same square footage of land. The bigger houses tend to be spread throughout equally with no obvious clustering
   * The rest of the more average sized houses are found everywhere regardless of the location
  * This means that the location itself of where houses are located must be the more important predictor of housing prices
   * We can reasonably infer that the Northern region is the downtown area of the county
  * 0.7 correlation between the size of the house and its grade
   * That is pretty reasonable, as a house would presumably be rated higher because of its larger housing unit
   
  ### Question 3: What combination of bathrooms, floors, and/or bedrooms indicates the higher price for houses? Are the findings significant enough?
  
  ![EDA_3](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/Capture_12.PNG)
  
  * Results indicate that a combination of all three variables yields the highest R-squared value
  * Interpretation: A r-squared value of 0.20 means that about 20% of the variance in our target variable ‘price’ is caused by our prediction model with bathrooms, bedrooms, and floors as the predictor variables
  * Generally, a R-squared value of 0.7 or higher is considered acceptable, so these three variables alone will not be enough for our regression model

### Check the three assumptions for linear regression

  * Linearity
  * Normality
  * Homoscedasticity

  *Iterate through all the predictor variables to check out if these three assumptions were met for any of them. Here is a sample of one of the graphs*
  
  ![scatter](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/linear.png)
  ![reg_exog](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/linear_regression_assumptions.png)
  ![qqplot](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/normality.png)
  
  ### Check for Multicollinearity
  
  The assumption in linear regression is that the dependent variable changes based on a change in an independent variable, with all other variables held constant. Now you can see why multicollinearnity may present and issue, as there may be overlapping effects on a target variable if two independent variables are highly correlated to one another.
  
  * Use a threshold correlation of >=0.7 & <=1.0 to observe overlapping variable effects
  * Use .stack() method to output the most highly correlated pair of predictor variables
  * Eliminate the variables sqft_living, sqft_above, sqft_living15, sqft_lot15, cond_3

  ![multicoll](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/multicollinearity.png)
  
  ### Use Stepwise selection with p-values to choose final features for the model
  
  **Chosen Features:**
    * lat
    * bedrooms 
    * grade 
    * floors 
    * bathrooms 
    * sqft_lot
    
   Scale these features afterwards to make each predictor variable's effects are calculated relatively in our final model
    
   ![chosen_features](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/final_model.PNG)
   
   ## Model Validation
   
   
  **Use Train-test split method: 80%-20% Training-Testing set split**
  
* Training set MSE: 12690209639.236786
* Testing set MSE: 12535566760.226467
* Neither underfitted nor overfitted

       
   ![test_data](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/training_test_error.png)

**Use Cross Validation with 5 Folds (better method)** 

* Train-test split MSE: 13055616350.428406
* Cross Val 5-Fold MSE: 12780931558.566036
* Confirms Train-test split method: model is not  underfitted nor overfitted


# Final Model

Our model will follow the following formula: *𝑦̂ = 𝛽̂0 + 𝛽̂1𝑥1+ 𝛽̂2𝑥2+ …+𝛽̂𝑛𝑥𝑛*

* 𝑦̂: "fitted line" or the predicted value associated with the predictor variables.
* 𝛽̂0: Intercept
* 𝛽̂1, 𝛽̂2: Coefficients of each selected predictor variable
* x1, x2: Predictor variables
* n: Number of predictors


**𝑦̂ = 224425.17(waterfront) + 145883.80(Lat) + 115566.39(Grade) + 47436.07(Bathrooms) + 12878.10(Bedrooms) - 15084887.70**

![model_graph](https://github.com/edwardcheng22/King-County-Housing-Prices-Prediction-Project/blob/master/Images/final_graph.png)


**Analysis**

* Bedrooms: Having a additional bedroom can bring a modest amount of $12878.10
* Grade: The grading assigned to a housing unit can drive up a house's price massively.  As seen in the EDA section, we can confirm this finding as we notice a high correlation between the price of a house and its grade. Grade brings in $115566.39 in value
* Lat: Can reasonably infer that being farther up North can add a huge amount of value – specifically $145883.80
* Bathrooms: An additional bathroom brings in $47436.07 in value, and this seems reasonable because most people would be satisfied with having more bathrooms.
* Waterfront: Brings in $224425.17, the highest valued predictor


   




  
