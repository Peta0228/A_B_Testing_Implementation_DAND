#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[114]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[115]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[116]:


df.shape[0]


# c. The number of unique users in the dataset.

# In[117]:


df.user_id.nunique()


# d. The proportion of users converted.

# In[118]:


df.query('converted == 1')['converted'].count() / df.converted.count()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[119]:


# Creating labels for differnt conditionings
new_page = df.landing_page == 'new_page'
new_page_F = df.landing_page != 'new_page'
treatment = df.group == 'treatment'
treatment_F = df.group != 'treatment'

# Use sets calculations to obtain the results
df.group[new_page].count() + df.group[treatment].count() -  2*(df.group[new_page & treatment].count())


# f. Do any of the rows have missing values?

# In[120]:


df.isnull().sum()


# In[121]:


df.isna().sum()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[122]:


df2 = df.drop(df[((new_page) & (treatment_F)) | ((new_page_F) & (treatment))].index,)


# In[123]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[124]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[125]:


# Creat a new column to identify the duplicated user_id
df2['uniqueness'] =  df2.user_id.duplicated()

df2.query('uniqueness == True')


# In[126]:


df2.query('user_id == 773192')


# c. What is the row information for the repeat **user_id**? 

# In[127]:


df2.query('uniqueness == True')


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[128]:


df2.drop(df2.query('uniqueness == True').index, inplace = True)


# In[129]:


# Drop the uniqueness column, since it is longer used now.
df2.drop(columns='uniqueness', inplace=True)


# In[130]:


df2.head()


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[131]:


df2.query('converted == 1')['converted'].count() / df2['converted'].count()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[132]:


# Creating labels for differnt conditionings
new_page = df2.landing_page == 'new_page'
new_page_F = df2.landing_page != 'new_page'
treatment = df2.group == 'treatment'
treatment_F = df2.group != 'treatment'


df2[treatment_F & df2['converted'] == 1]['converted'].count() / df2[treatment_F]['converted'].count()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[133]:


df2[treatment & df2['converted'] == 1]['converted'].count() / df2[treatment]['converted'].count()


# d. What is the probability that an individual received the new page?

# In[134]:


df2.query('landing_page == "new_page"')['converted'].count() / df2['converted'].count()


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# No, there is no sufficient evidence to suggest an increase in conversions from the change in landing page, as the stats were not pratically siginificant enough. 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **Null hypothesis: (P_new - P_old) <= 0;
# Alternative hypothesis: (P_new - P_old) > 0**

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[135]:


P_new = df2.query('converted == 1')['converted'].count() / df2['converted'].count()


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[136]:


P_old = df2.query('converted == 1')['converted'].count() / df2['converted'].count()


# c. What is $n_{new}$?

# In[137]:


n_new =  df2[new_page]['converted'].count()
n_new


# d. What is $n_{old}$?

# In[138]:


n_old = df2[new_page_F]['converted'].count()
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[139]:


new_page_converted = np.random.choice([0,1], size=n_new, p=[1-P_new, P_new])


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[140]:


old_page_converted = np.random.choice([0,1], size=n_old, p=[1-P_old, P_old])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[141]:


P_diff = (new_page_converted.sum() / len(new_page_converted)) - (old_page_converted.sum() / len(old_page_converted))


# In[142]:


P_diff


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[143]:


p_diffs = []

# Simluate the 10,000 iterations
for i in range(int(1e4)):
    new_page_converted = np.random.choice([0,1], size=n_new, p=[1-P_new, P_new])
    old_page_converted = np.random.choice([0,1], size=n_old, p=[1-P_old, P_old])
    P_diff = (new_page_converted.sum() / len(new_page_converted)) - (old_page_converted.sum() / len(old_page_converted))
    p_diffs.append(P_diff)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[144]:


plt.hist(p_diffs);


# **The plot looks like normal distrubution, it is what I expected baseed on the central limit theorem.**

# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[145]:


# Calculate the actual difference observed before 
old = df2[treatment_F & df2['converted'] == 1]['converted'].count() / df2[treatment_F]['converted'].count()
new = df2[treatment & df2['converted'] == 1]['converted'].count() / df2[treatment]['converted'].count()
obs_diff = new - old


# In[146]:


obs_diff


# In[147]:


plt.hist(p_diffs);
plt.axvline(obs_diff, c='red');


# In[148]:


# The proportion that p_diffs is great than the actual difference
(p_diffs > obs_diff).mean()


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **This is called the p value in scientific studies, it means the probability of observing our statistic or a more extreme statistic from the null hypothesis. In this case, there is 90.26% of chance for us to observe the alternative hypothesis, which is (P_new - P_old) > 0, when the null hypothesis (P_new - P_old) <= 0 is true. Therefore we fails to reject the null hypothesis.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[149]:


import statsmodels.api as sm

convert_old = df2[treatment_F & df2['converted'] == 1]['converted'].count()
convert_new = df2[treatment_F & df2['converted'] == 1]['converted'].count()
n_old = df2[treatment_F]['converted'].count()
n_new = df2[treatment]['converted'].count()


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[162]:


# the method is used based on the null hypothesis that old page is at least as goof as the new page in terms of conversion rate
sm.stats.proportions_ztest(count=[convert_old,convert_new], nobs=[n_old,n_new], alternative='smaller')


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **they mean that there is a chance of 50.99% of observing a stastistics for alternative, when the null is actually true. Therefore, we fail to reject the null hypothesis, which agrees with the previous findings.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic regression.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[151]:


df2['intercept'] = 1
df2[['ab_page', 'drop']] = pd.get_dummies(df.landing_page)


# In[152]:


df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[153]:


lm = sm.Logit(df2.converted, df2[['intercept', 'ab_page']])
mod = lm.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[154]:


mod.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **The p-value associated with ab_page is 0.190, which makes ab_page not statistically significant enougth to predict the conversion. According to http://www.biostathandbook.com/multiplelogistic.html, the null hypothesis associated with the logistic regressoin moedel is: there is no relationship between the two variables; while the alternative hypothesis being: there is a relationship between the two variables. Therefore, the two hypothesises are fundamentally different from the two in Part II, and so their vlaues also differed from Part II.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Because there might be one or more factors affecting the Y variable. However, addtional terms may arise problems like multi-collinearity, which means the two explanatory variables are correlated.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[155]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')


# In[156]:


df_new.head()


# In[157]:


### Create the necessary dummy variables
df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new.country)


# In[158]:


df_new.drop(columns=['country', 'timestamp', 'drop'], inplace=True )


# In[159]:


df_new.head()


# In[160]:


lm = sm.Logit(df_new.converted, df_new[['intercept','CA', 'UK']])
mod = lm.fit()
mod.summary()


# **As from the above summary, none of the variables are statistically significant enough to expian the converision. Therefore, country does not appear to have an impact on conversion.**

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[161]:


### Fit Your Linear Model And Obtain the Results
lm = sm.Logit(df_new.converted, df_new[['intercept','CA', 'UK', 'ab_page']])
mod = lm.fit()
mod.summary()


# **Similarly,  none of the variables are statistically significant enough to expian the converision.**

# <a id='conclusions'></a>
# ## Conclusions
# 
# **Conclusively speaking, I suggest the company to keep the old page. As from Part II, the A/B test fails to reject the null hypothesis, which says the old page is at least as great as the new page in terms of conversion rate. Therefore, keeping the old page would be a resonable chocie for the company. However, the regression models did not successfully draw a relationship between the two pages, as well as the influence of nationality to conversion, this may indicate that the company should continue to run the epxeriment, in order to obtain a model that is statistically significant enough.**
