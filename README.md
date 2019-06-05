# Executive Summary

## Problem Statement

   Reddit aggregates some of the best content in the world. If you see it on Facebook, Twitter, or LinkedIn there is a high possibility it originated on Reddit first. The site isn't just a hodge podge of virtually every community on the planet talking about issues/musings that are on their mind, there is some pretty tight moderation to keep things under control and running smoothly. However, Imagine a scenario where the Reddit servers are down, or Reddit gets hacked, and people's message history gets disorganized. Does Reddit have a plan in place to solve it? How would they mitigage this potential disaster? Based off the post titles and its references, can we predict where a post comes from using Natural Language Processing (NLP) and Classification Models? Yes. 

   From here moving into the future, managing your information and cyberthreats are one of the top risk factors corporations are concerned about. It would mean a huge interruption to core operations. Even the very largest organizations and governments are not immune to it, and those that are susceptible to it can only hope to contain it. It gets worse when you think about how computing power, AI, machine learning, and mobile device usage are starting to outpace the protections companies have in place currently. The risk for disruption from within or externally increases exponentially. Even if the possibility of a scenario like this is small, a number of successive disruptions like this could put a dent in Reddit's popularity. Reddit needs a way to quickly reorient themselves if information got disorganized. This is how you survive in the future.  

## Solution

   To address this problem we are going to create a classification model and utilize NLP. This is necessary because what we want to do is determine the class that a particular post resides in. In this case the difference between two subreddits. As an example we can use a test case of r/Politics and r/Stocks after pulling down the data using beautiful soup to place it in a dataframe. The two subreddits have an implicit relationship at the cross section of government policy and economics. NLP will help us to turn our everyday language into numbers, allowing the computer to interpret and codify the information to achieve our desired result. The objective of the classification model is to identify, based off data we feed it, which category the information will fall under. 

The types of classification models include: 
- Logistic Regression
- Naive Bayes (Guassian, Binomial, and Multinomial)
- Decision Trees
- Random Forests
- K-Nearest Neighbors
- Support Vector Machines

Each has it's own unique way for identifying the class that the data resides in. I will be using Logistic Regression, Multinomial Naive Bayes, and Random Forests(I will detail each in the modeling section of the jupyter notebook). Logistic Regression is the least complex in terms of what it needs to execute while Random Forest needs alot and is the most computationally expensive of the 3. After modeling it turned out that the least complex was 95% accurate in determining where a reddit post comes from.

## Methodology

The algorithm that I created stripped the title text of any unnecessary words, characters, and any unnecessary text. I also used grid search to find the optimal parameters for the model to use to achieve the best score. Once the best parameters are found I used those same parameters to print out the top score of the model, so every model presented here is the best based off the parameters and boundaries I provided it. The parameters I decided to stick with were basic to ensure the models weren't too overwhelming for the CPU. The logic behind the choices made for the parameters (this applies to all models) is I wanted to make sure there was a boundary in the minimum amount of times a word needed to appear in the document, and the maximum amount of times it could be a document. Then I also wanted to set a limit the number of max features for the models. Although it should theoretically increase the performance this is not the case for Random Forest as it decreases the diversity of the trees. Also increasing the feature amount would cause the speed to execution to decrease for the model. To address any overfitting, where our model fits the data almost perfectly, I supplied L1 (Lasso) and L2 (Ridge) penalties.

## Conclusion

In conclusion, the least complex model, the Logistic Regression Model with Countvectorization, beat out all of the rest for both training and testing with 95% accuracy. If I had more time I would have played with more parameters and allowed to models to take time to run through a combonation of different parameters to find the true optimal model for each. Also, during EDA I discovered that where was not much overlap between the two subreddits, which made it easy for the model to predict which subreddit a post comes from. I recommend in the future trying model for a very explicit relationship between two subreddits that have alot of overlap. 
<<<<<<< HEAD

## Resources

1. http://localhost:8888/notebooks/DSI%20-%20Nash/GALessons/5_Week/5.06-lesson-nlp_ii/introduction-to-nlp.ipynb
2. http://localhost:8888/notebooks/DSI%20-%20Nash/GALessons/5_Week/5.07-lesson-naive_bayes/starter-code.ipynb
3. https://www.youtube.com/watch?v=iQ1bfDMCv_c -- Majority of credit is due to Alice Zhao on youtube who does a walkthrough of Natural Language Processing from beginning to end. 
4. https://www.analyticsindiamag.com/7-types-classification-algorithms/
=======
>>>>>>> 96df7be221ab27751f9f6e96b365d294a2d22002
