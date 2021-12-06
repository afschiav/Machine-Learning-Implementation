#####TABLE OF CONTENTS####

#REGRESSION ANALYSIS
#-CROSS SECTIONAL ESTIMATION
#---OLS
#---GLS
#---Logistic
#---Ridge
#---Lasso
#---Elastic-Net
#---Diagnostics
#-TIME SERIES ESTIMATION
#---Decompositions
#---Stationarity
#---Forecasting and Smoothing
#---AR and ARIMA models
#-PANEL ESTIMATION
#-SYSTEM OF EQUATIONS

#DIMENSION REDUCTION
#-Principal Component Analysis (PCA)
#-Multidimensional Scaling (MDS) & Principal Coordinate Analysis (PCoA)
#-t-SNE

#MIXTURE MODELS
#-Univarite
#-Multivariate

#DECISION TREES & RANDOM FORESTS
#-Decision Tress
#-Random Forests
#-AdaBoost

#####REGRESSION ANALYSIS####

#CROSS SECTIONAL DATA
{
  
  #Resources
  {
    #Difference between GLS and OLS: https://stats.stackexchange.com/questions/155031/how-to-determine-if-gls-improves-on-ols 
    #Testing for mutlicollinearity: https://datascienceplus.com/multicollinearity-in-r/
    #Testing for heteroskedasticity: https://rpubs.com/cyobero/187387
    #Ridge regression: https://www.youtube.com/watch?v=Q81RR3yKn30&ab_channel=StatQuestwithJoshStarmer
    #Cross validation: https://www.youtube.com/watch?v=fSytzGwwBVw&ab_channel=StatQuestwithJoshStarmer
    #Ridge, Lasso, and Elastic-Net Regression in R:  https://www.youtube.com/watch?v=ctmNq7FgbvI&ab_channel=StatQuestwithJoshStarmer 
    #Logistic regression in R: https://www.youtube.com/watch?v=C4N3_XJJ-jU&ab_channel=StatQuestwithJoshStarmer
    }
  
  #OLS
  {
    #Libraries
    library(stats) #necessary for 'lm'
    
    #Data
    data(iris)
    View(iris)#for some reason you need to run built in data through a fuction to get it to show up in global environment 
    iris<-as.data.frame(iris)#Convert to dataframe 
    
    #estimate and summarize model
    model_ols<-lm(data=iris, Petal.Length~Sepal.Length+Sepal.Width)
    summary(model_ols) 
    
    #clear global environment
    rm(list=ls())
  }

  #GLS
  {
    #Libraries
    library(nlme) #necessary for 'gls'
    
    #Data
    data(iris)
    View(iris)#for some reason you need to run built in data through a fuction to get it to show up in global environment 
    iris<-as.data.frame(iris)#Convert to dataframe 
    
    #estimate and summarize model
    model_gls<-gls(data=iris, Petal.Length~Sepal.Length+Sepal.Width)
    summary(model_gls) 
    
    #NOTE:The difference between OLS and GLS is the assumptions made about the error term
    #of the model. In OLS we (at least in CLM setup) assume that Var(u)=\sigma2I, where I is the 
    #identity matrix such that there are no off diagonal elements different from zero. 
    #With GLS this is no longer the case (it could be, but then GLS = OLS). With GLS we assume
    #that Var(u)=\sigma2\Theta, where \Theta is the variance-covariance matrix.
    
    #clear global environment
    rm(list=ls())
  }
  
  #Logistic regression
  {
    #Technique to compute probability of membership in group; often used for classification (i.e. p>0.5 => Type 1)
    #Logistic regression uses Maximum likelihood to estimate parameters
    #Logistic regression is a specific type of Generalized Linear Models (GLM)
    #Y-axis is confined to p in [0,1]. This is transformed to the (-inf,inf) number line using log odds function ("logit"): log(p/(1-p))
    #Logit transformation: 0 => -inf, 0.5 => 0, 1 => inf
    #Coefficients of logistic regression are presented using the log-odds transformation (meaning they are not interpreted as probabilities)
    #MLE estimation iterates through candidate lines in log-odds space, transforms proections to probability (called likelihood in this case) and compute for data. Then rotate candidate line
    
    #Libraries
    library(ggplot2)
    library(cowplot)
    
    #Example
    
    #Get data
    url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    data <- read.csv(url, header=FALSE)
    
    #Reformat data
    {
      #####################################
      ##
      ## Reformat the data so that it is
      ## 1) Easy to use (add nice column names)
      ## 2) Interpreted correctly by glm()..
      ##
      #####################################
      head(data) # you see data, but no column names
      
      colnames(data) <- c(
        "age",
        "sex",# 0 = female, 1 = male
        "cp", # chest pain
        # 1 = typical angina,
        # 2 = atypical angina,
        # 3 = non-anginal pain,
        # 4 = asymptomatic
        "trestbps", # resting blood pressure (in mm Hg)
        "chol", # serum cholestoral in mg/dl
        "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
        "restecg", # resting electrocardiographic results
        # 1 = normal
        # 2 = having ST-T wave abnormality
        # 3 = showing probable or definite left ventricular hypertrophy
        "thalach", # maximum heart rate achieved
        "exang",   # exercise induced angina, 1 = yes, 0 = no
        "oldpeak", # ST depression induced by exercise relative to rest
        "slope", # the slope of the peak exercise ST segment
        # 1 = upsloping
        # 2 = flat
        # 3 = downsloping
        "ca", # number of major vessels (0-3) colored by fluoroscopy
        "thal", # this is short of thalium heart scan
        # 3 = normal (no cold spots)
        # 6 = fixed defect (cold spots during rest and exercise)
        # 7 = reversible defect (when cold spots only appear during exercise)
        "hd" # (the predicted attribute) - diagnosis of heart disease
        # 0 if less than or equal to 50% diameter narrowing
        # 1 if greater than 50% diameter narrowing
      )
      
      head(data) # now we have data and column names
      
      str(data) # this shows that we need to tell R which columns contain factors
      # it also shows us that there are some missing values. There are "?"s
      # in the dataset. These are in the "ca" and "thal" columns...
      
      ## First, convert "?"s to NAs...
      data[data == "?"] <- NA
      
      ## Now add factors for variables that are factors and clean up the factors
      ## that had missing data...
      data[data$sex == 0,]$sex <- "F"
      data[data$sex == 1,]$sex <- "M"
      data$sex <- as.factor(data$sex)
      
      data$cp <- as.factor(data$cp)
      data$fbs <- as.factor(data$fbs)
      data$restecg <- as.factor(data$restecg)
      data$exang <- as.factor(data$exang)
      data$slope <- as.factor(data$slope)
      
      data$ca <- as.integer(data$ca) # since this column had "?"s in it
      # R thinks that the levels for the factor are strings, but
      # we know they are integers, so first convert the strings to integers...
      data$ca <- as.factor(data$ca)  # ...then convert the integers to factor levels
      
      data$thal <- as.integer(data$thal) # "thal" also had "?"s in it.
      data$thal <- as.factor(data$thal)
      
      ## This next line replaces 0 and 1 with "Healthy" and "Unhealthy"
      data$hd <- ifelse(test=data$hd == 0, yes="Healthy", no="Unhealthy")
      data$hd <- as.factor(data$hd) # Now convert to a factor
      
      str(data) ## this shows that the correct columns are factors
      
      ## Now determine how many rows have "NA" (aka "Missing data"). If it's just
      ## a few, we can remove them from the dataset, otherwise we should consider
      ## imputing the values with a Random Forest or some other imputation method.
      nrow(data[is.na(data$ca) | is.na(data$thal),])
      data[is.na(data$ca) | is.na(data$thal),]
      ## so 6 of the 303 rows of data have missing values. This isn't a large
      ## percentage (2%), so we can just remove them from the dataset
      ## NOTE: This is different from when we did machine learning with
      ## Random Forests. When we did that, we imputed values.
      nrow(data)
      data <- data[!(is.na(data$ca) | is.na(data$thal)),]
      nrow(data)
      
      #####################################
      ##
      ## Now we can do some quality control by making sure all of the factor
      ## levels are represented by people with and without heart disease (hd)
      ##
      ## NOTE: We also want to exclude variables that only have 1 or 2 samples in
      ## a category since +/- one or two samples can have a large effect on the
      ## odds/log(odds)
      ##
      ##
      #####################################
      xtabs(~ hd + sex, data=data)
      xtabs(~ hd + cp, data=data)
      xtabs(~ hd + fbs, data=data)
      xtabs(~ hd + restecg, data=data)
      xtabs(~ hd + exang, data=data)
      xtabs(~ hd + slope, data=data)
      xtabs(~ hd + ca, data=data)
      xtabs(~ hd + thal, data=data)
      
    }
    
    #Simple model (univariate):
    logistic<-glm(hd ~ sex, data=data, family = "binomial")
      #Use glm() for Generalized Linear Models
      #Write formulta
      #family="binominal' => tells glm() to do logistic regression
    summary(logistic)
      #Want "Deviance Residuals" to be centered around zero and roughly symmetrical. These look good!
      #Coefficient for sexM is interpretted as the increase in the log(odds) that a male has of having heart disease
      #Std. error, z-values, and p-values are results of Wald's test 
    
    #Alternatvie model (all variables used):
    logistic<-glm(hd ~ ., data=data, family="binomial")
    summary(logistic)
      #REsidual Deviance and AIC are much smaller for this model, meaning it is a better model
    
    #Compute McFadden's Pseudo R^2:
    ll.null<-logistic$null.deviance/-2 #Get log-likelihood of the null model
    ll.proposed<-logistic$deviance/-2 #Get log-likelihood of the proposed model
    (ll.null-ll.proposed)/ll.null #Get pseudo R^2
    
    #Graph predicted probabilities:
    predicted.data<-data.frame(probability.of.hd=logistic$fitted.values, hd=data$hd)#create new df w/ probs. of having heart disease along with actual heart disease status
    predicted.data<-predicted.data[order(predicted.data$probability.of.hd, decreasing=FALSE),]#sort by probs from low to high
    predicted.data$rank<-1:nrow(predicted.data)#add rank from low to high prob.
    
    ggplot(data=predicted.data, aes(x=rank, y=probability.of.hd)) +
      geom_point(aes(color=hd), alpha=1, shape=4, stroke=2) +
      xlab("Index") +
      ylab("Predicted probability of getting heart disease")
    
    rm(list=ls())
    
  } 
  
  #Ridge Regression
  { 
    #Libraries
    library(glmnet)#necessary for ridge, lasso, and elastic-net regression
    
    #-Ridge regression minimizes the sum of squared residuals + lambda*slope^2 (lambda is a penalty). In general, the ridge regression penalty contains all regression parameters except intercept.
    #-The purpose of Ridge regression is to introduce some bias (worse fit to training data) to reduce variance in real data
    #-To select lamda, try a bunch of values and use cross validation to determine which minimizes variance.
    #-Ridge regression allows for estimating models with # of parameters> # of data points
  
    #Example
    
    #generate data:
    set.seed(42)
    n<-1000 #number of obs.
    p<-5000 #number of parameters
    real_p<- 15 #number of important parameters
    x<-matrix(rnorm(p*n), nrow=n, ncol=p) #generate data
    y<-apply(x[,1:real_p], 1, sum)+rnorm(n) #generate vector of columns that depend on the first 15 columns of x plus some noise
    
    train_rows<-sample(1:n, .66*n) #randomly select 2/3rds of data for training set
    
    x.train<-x[train_rows,] #training data
    x.test<-x[-train_rows,] #testing data
    
    y.train<-y[train_rows] #training data
    y.test<-y[-train_rows] #testing data
    
    alpha0.fit<-cv.glmnet(x.train, y.train, type.measure = "mse", alpha=0, family="gaussian") #estimate Ridge model
      #cv=cross validation (default ten fold) for parameter. 
      #type.measure=mean squared error (criteria for cross-validation). For logistic, set equal to "deviance"
      #alpha=0 => Ridge regression (note that R uses weighted average to mix Ridge and Lasso)
    
    alpha0.predicted<-predict(alpha0.fit, s=alpha0.fit$lambda.1se, newx = x.test)#Apply Ridge model to testing data
      #s ("size of penatly") is set equal to optimal value of lambda. Could also set equal to lambda.min, which is the lambda that resulted in smallest sum
      #newx is the test data
    
    mean((y.test-alpha0.predicted)^2) #Compute mean squared error between true vales and predicted values
  }   
   
  #Lasso Regression
  {
    #Lasso regression minimizes the sum of squared residuals + lambda*|slope|. 
    #Big difference between Lasso and Ridge regression: while Ridge can only shrink slope asmptotically close to zero,
    #Lasso can shrink slope all the way to zero. This helps eliminate the impact of unimportant variables in the regression
    
    
    
    #Example (Note: Must load data in 'Ridge Regression' section)

    alpha1.fit<-cv.glmnet(x.train, y.train, type.measure = "mse", alpha=1, family="gaussian") #estimate Lasso model
      #alpha=1 => Lasso regression
    
    alpha1.predicted<-predict(alpha1.fit, s=alpha1.fit$lambda.1se, newx = x.test)#Apply Lasso model to testing data
    
    mean((y.test-alpha1.predicted)^2) #Compute mean squared error between true vales and predicted values
    
  }
  
  #Elastic-Net Regression
  {
    #Elastic-Net combines the Ridge and Lasso penalties such that it minimizes:
    #        SSR+lambda1*(variable1^2+variable2^2+...)+lambda2*(|variable1|+|variable2|+...)
    #Use cross validation on combinantions of lamda1 and lambda2 to find the best values.
    #particularly useful when there is multicollinearity in the model
    
    #Example (Note: Must load data in 'Ridge Regression' section)
    
    alpha0.5.fit<-cv.glmnet(x.train, y.train, type.measure = "mse", alpha=0.5, family="gaussian") #estimate Elastic-Net model
    #alpha=0.5 => Elastic-Net regression with even mix of Ridge and Lasso
    
    alpha0.5.predicted<-predict(alpha0.5.fit, s=alpha0.5.fit$lambda.1se, newx = x.test)#Apply Elastic-Net model to testing data
    
    mean((y.test-alpha0.5.predicted)^2) #Compute mean squared error between true vales and predicted values
    
    #NOTE: 0.5 is only one possible mixture of Ridge and Lasso. What about others?
    list.of.fits<-list()
    
    for(i in 0:10){ #loop through values of 0 to 1 by 0.1 increments...
      fit.name=paste0("alpha", i/10)
      
      list.of.fits[[fit.name]]<-
        cv.glmnet(x.train, y.train, type.measure = "mse", alpha=i/10, family="gaussian") #...and estimate Elastic-Net with alpha=i
    }
    
    results<-data.frame()
    
    for (i in 0:10){  #loop through values of 0 to 1 by 0.1 increments...
      fit.name=paste0("alpha", i/10)
      
      predicted<-predict(list.of.fits[[fit.name]],
                         s=list.of.fits[[fit.name]]$lambda.1se,
                         newx=x.test) #...and predict using Elastic-Net with alpha=i
      mse<-mean((y.test-predicted)^2) #...and compute mse
      
      temp<-data.frame(alpha=i/10, mse=mse, fit.name=fit.name)
      results=rbind(results, temp) #bind to results
    }
    
    #print best model (minimum MSE)
    results[which.min(results$mse),]
    
    #clear global environment
    rm(list=ls())
  
  }
  
  #Diagnostics
  {
    #Libraries
    library(corpcor) #necessary for cor2pcor
    library(ggplot2) #plot residuals
    
    #Data
    data(iris)
    View(iris)#for some reason you need to run built in data through a fuction to get it to show up in global environment 
    iris<-as.data.frame(iris)#Convert to dataframe 
    
    #Check for multicollinearity:
    covariates<-iris %>% select(Sepal.Length,Sepal.Width)
    cor2pcor(cov(covariates)) #cor2pcor computes the pairwise partial correlation coefficients from either a correlation or a covariance matrix.
    #pairwise comparisons compute the mean difference of each pair
    
    #Check for heteroskedasticity (Visual):
    iris$residual<-model_ols$residuals
    ggplot(data = iris, aes(y = residual, x = Sepal.Length)) + geom_point(col = 'blue') + geom_abline(slope = 0) #plot residuals vs. dependent variables
    ggplot(data = iris, aes(y = residual, x = Sepal.Width)) + geom_point(col = 'blue') + geom_abline(slope = 0) #plot residuals vs. dependent variables
    
    #Breusch-Pagan Test for heteroskedasticity:
    var.func <- lm(residual^2 ~ Sepal.Length, data = iris)#regress dependent variable on squared residuals
    summary(var.func)#R^2 multiplied by N gives test statistic
    t_stat<-0.009322*150 #compute statistic
    t_crit<-qchisq(0.95, 1)#obtain critical value
    ifelse(t_stat<t_crit, ("No heteroskedasticity"), ("Possible heteroskedasticity"))
    
    #clear global environment
    rm(list=ls())
    
  }
  
}

#TIME SERIES DATA
{
  #Resources
  {
    #Using R for time series analysis: https://a-little-book-of-r-for-time-series.readthedocs.io/en/latest/src/timeseries.html 
    #Holt Winters forecasting: https://orangematter.solarwinds.com/2019/12/15/holt-winters-forecasting-simplified/
    #Stationarity testing: https://rpubs.com/richkt/269797
    #ARIMA modeling: https://otexts.com/fpp2/arima-r.html 
  }
  
  #Libraries
  library("TTR") #necessary for smooting function
  library(mFilter)#necessary for HP filter
  library("forecast") #necessary for Holt Winters forecast
  library(tseries)#necessary for ADF test
  
  
  #Data:
  data(LakeHuron)
  LakeHuron<-as.data.frame(LakeHuron)#Convert to dataframe 
  LakeHuron<-LakeHuron %>% rename(lake_level=x)
  
  #Convert to time-series oject
  LakeHuron<-ts(LakeHuron, start=1875)
  plot.ts(LakeHuron) #plot time series
  
  #Decomposing trend and cycle
  LakeHuronSMA5<-SMA(LakeHuron,5)#compute 5-year moving average
  LakeHuronHP<-hpfilter(LakeHuron, type="lambda", freq = 6.25, drift=FALSE) #apply HP filter - Note: The larger the value of {\displaystyle \lambda }\lambda , the higher is the penalty. Hodrick and Prescott suggest 1600 as a value for {\displaystyle \lambda }\lambda  for quarterly data. Ravn and Uhlig (2002) state that {\displaystyle \lambda }\lambda  should vary by the fourth power of the frequency observation ratio; thus, {\displaystyle \lambda }\lambda  should equal 6.25 (1600/4^4) for annual data 
  plot.ts(cbind(LakeHuron,LakeHuronSMA5, LakeHuronHP$trend, LakeHuronHP$cycle)) #plot original, MA, HP trend, HP cycle

  #Testing for stationarity
  acf(LakeHuron) #plot autocorrelation function
  Box.test(LakeHuron, lag=10, type="Ljung-Box") #Ljung-Box test for stationarity at given 1-10 lags. H0=sationarity
  adf.test(LakeHuron) #Augmented Dickey-Fuller t-test for unit root, H0=unit root
  kpss.test(LakeHuron, null="Trend") #Kwiatkowski-Phillips-Schmidt-Shin (KPSS) for level or trend stationarity, test the null hypothesis of trend stationarity
  acf(diff(LakeHuron)) #plot ACF of first-difference series (stationary)
  
  #Forecasting using exponential smoothing
  LakeHuronHW<-HoltWinters(LakeHuron, beta=FALSE, gamma=FALSE)#Exponential smoothing: exponentially weighted moving average (EWMA) to "smooth" a time series
  print(LakeHuronHW_forecast)#The value of alpha; lies between 0 and 1. Values of alpha that are close to 0 mean that little weight is placed on the most recent observations when making forecasts of future values, values close to 1 mean latest values have more weight.
  plot(LakeHuronHW_forecast)#plot fitted and actual values
  Forecast_LakeHuronHW<-forecast:::forecast.HoltWinters(LakeHuronHW, h=4)#forecast four years using HW. Note triple ":::" necessary here to return the value of internal variable
  plot(Forecast_LakeHuronHW)#plot forecast
  
  #AR and ARIMA estimation
  AR<-ar(LakeHuron, aic=TRUE) #fit an AR model using AIC to determine lags
  predict(AR, n.ahead=10) #Use AR model to predict 10 years ahead
  ARIMA<-arima(LakeHuron, order=c(3,1,1))#fit ARIMA model with three lags, I=1, 1 MA component
  predict(ARIMA, n.ahead=10) #Use ARIMA model to predict 10 years ahead
  
  
  }  

#PANEL DATA
{
  
}


#####DIMENSION REDUCTION####

#Resources
{
  #Prinicapl cmponent analysis: https://www.youtube.com/watch?v=FgakZw6K1QQ&ab_channel=StatQuestwithJoshStarmer
}


#Principal Component Analysis (PCA)
{
  #PCA reduces data with 3 or more dimensions into a two-dimensional space
  #It can tell us which variable is most valuble for clustering data
  
  #Step 1: compute average of data by finding average of all variables
  #Step 2: Recenter the data around the center of the data such that center of data is now point (0,0,....,0)
  #Step 3: Find best fitting line to recentered data. This is done by maximizing the sum of squared distances from the projected points on the line to the origin. These distances are called Eigenvalues
  #        Mathematically this process is the same as minimizing the distance between the points and the line. 
  #Step 4: This is called PC1, which is a linear combination of two variables (gives cocktail mix of two variables)
  #Step 5: scale linear combination so that the hypotenuse is of length 1, creating unit vector (Eigenvector for PC1). Proportions of each variables are called 'loading scores'
  #Step 6: Find PC2. With only two variables, it is just the unit vector that is perpendicular to Eigenvector of PC1.
  #Step 7: Rotate data such that PC1 and PC2 are (1,0) and (0,1), respectively
  #Step 8: Compute variation for each PC:
  #            Variation for PC1=SS(distance for PC1)/(n-1)
  #            Variation for PC2=SS(distance for PC2)/(n-1)
  #Step 9: Calculate percentage of variaton for each PC:
  #            Percentage of variation for PC1=Variation for PC1/(Variation for PC1+Variation for PC2)
  #            Percentage of variation for PC2=Variation for PC2/(Variation for PC1+Variation for PC2)
  #         #Note: a Scree plot is a graphical representation of the percentage of variation that each PC accounts for.
  
  
  #PCA with 3+ variables is nearly the same, adding additional steps after Step 6. In theory, there is one PC per variable, but in practice the number of PCs is either the number of variables
  #or the number of samples, whichever is less. 
  
  #Using Scree plot, we can select the one or two most important PC(s) and plot them. This gives visualization of data clusters in lower dimensions!
  
  
}



#####MIXTURE MODELS####

#Libraries
library(mclust)
library(dplyr)

#Resources
{
  #https://www.youtube.com/watch?v=DODphRRL79c&t=574s&ab_channel=MachineLearningTV
  #https://cran.r-project.org/web/packages/mclust/vignettes/mclust.html
  #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5096736/
}

#Data
data(mtcars)

#UNIVARIATE MMs:
{
#model (1):
univar<-Mclust(mtcars$mpg) #run on 'mpg' variable, no restrictions
summary(univar) #summarize
plot(univar, what="BIC")#show BIC selection process
plot(univar, what="classification")#classification
plot(univar, what="density")#guassian densities
plot(univar, what="uncertainty")#classification uncertainty

#model (2):
univar<-Mclust(mtcars$mpg, G=2) #run on 'mpg' variable, restrict to two groups
summary(univar) #summarize
plot(univar, what="BIC")#show BIC selection process
plot(univar, what="classification")#classification
plot(univar, what="density")#guassian densities
plot(univar, what="uncertainty")#classification uncertainty


#model (3):
univar<-Mclust(mtcars$mpg, G=2, modelNames = "E") #run on 'mpg' variable, restrict to two groups, specify modeltype "E" for equal variance
summary(univar) #summarize
plot(univar, what="BIC")#show BIC selection process
plot(univar, what="classification")#classification
plot(univar, what="density")#guassian densities
plot(univar, what="uncertainty")#classification uncertainty

rm(univar)
}

#MULTIVARIATE MMs:
{
  #model (1):
  multivar<-Mclust(mtcars %>% select(mpg, cyl)) #two dimensional gaussian mixture model
  summary(multivar)
  plot(multivar, what="BIC")
  plot(multivar, what="classification")
  plot(multivar, what="uncertainty")
  
  #model (2):
  multivar<-Mclust(mtcars %>% select(mpg, cyl), G=3, modelNames = "EII") #two dimensional gaussian mixture model with two components, spherical distribution, equal volume and equal shape
  summary(multivar)
  plot(multivar, what="BIC")
  plot(multivar, what="classification")
  plot(multivar, what="uncertainty")
  
  rm(list=ls())
}

#####DECISION TREES & RANDOM FORESTS####

#Resources
{
  #Decision trees:https://www.youtube.com/watch?v=7VeUPuFGJHk&ab_channel=StatQuestwithJoshStarmer
  #Decision trees in R: https://www.youtube.com/watch?v=uXIIk7suD6c&ab_channel=DataDaft 
  #Random forests: https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&ab_channel=StatQuestwithJoshStarmer
  #Random forests in R: https://www.youtube.com/watch?v=6EXPYzbfLCE&ab_channel=StatQuestwithJoshStarmer 
  #AdaBoost: https://www.youtube.com/watch?v=LsK-xG1cLYA&ab_channel=StatQuestwithJoshStarmer 
}

#Decision trees
{
  #Decision trees classify people based on questions. Start at top, work your way down.
  #Top of tree is called "Root node". Leafs are called "leaf nodes".
  #A separation is considered "impure" if it does not correctly categorize people with 100% accuracy.
  #To determine which separation is best, need a way to measure impurity: "Gini Impurity".
  #For a given leaf node, Gini impurity is defined as: 1-(probability of TRUE)^2-(probability of FALSE)^2
  #For a separation with k leaves, compute weighted average of leaf node impurities (weights are number of people in each leaf)
  #Separations with the lowest impurity should be at root of tree. Then recalculate impurity of remaining separations
  #If Gini impurity of a node is lower than a potential separation, do not add separation, make node a leaf.
  #Algorithm summary:
  #   1) Calculate all of the Gini impurity scores
  #   2) If the node itself has the lowest score, then there is no point separating further and it becomes a leaf node
  #   3) If separating the data results in an improvement, then pick the separation with the lowest impurity value
  #
  #NOTE: If using continuous values, calculate the middle value between each ranked observation and compute its Gini impurity.
  #      Then select the threshold value with the lowest impurity value as the separation
  
  
  #Example: Survival prediction
  
  #Libraries
  library(rpart) #necessary for decision trees
  library(rpart.plot) #for plotting decision trees (prp)

  #Data
  data("kyphosis")
  view(kyphosis)
  
  simple_tree <- rpart(Kyphosis ~  Number + Start, data = kyphosis) #Build tree based on the following variabes.
    #Since Age, Number, and Start are all numeric, algorithm first determines optimal separation threshold
    #Note: order of variables does not matter, since these are optimized using the Gini impurity method
  
  prp(simple_tree) #Plot decision tree
  
  
  complex_tree <- rpart(Kyphosis ~ Age + Number + Start,
                        cp=0.001, #complexity parameter (tells algorithm when to stop)
                        data = kyphosis) #Build tree based on the following variabes.
  prp(complex_tree) #Plot decision tree
  
  
  
  rm(list=ls())

}

#Random forests
{
  #BASICS:
  
  #Random forests are built from decision trees
  #Idea is to create a data type that is more flexibile to new data
  #
  #Steps for building random forest:
  #1) Create a bootstrapped data set (randomly select samples from original data set, allowing same sample to be picked more than once)
  #2) Create a decision tree using bootstrapped dataset, but only use a random subset of (k<N) variables at each step. See later for optimal k
  #3) Repeat many times.
  
  #After repeating this process hundreds of time, you have hundreds of decision trees for classifying data.
  #When classifying new data, run down all trees and compute the number of votes for type, type with most votes is selected.
  #"Bootstrapping" data + "aggregating results" = "bagging".
  #Data that is not selected in boostrap is called "out of bag" dataset.
  
  #Performance of random forest can be tested by running out of bag samples and seeing how many are correctly classified
  #The proportion of out of bag samples that were incorrectly classified is the "out-of-bag error".
  
  #The number of variables per set (k) can be chosen by minimizing out-of-bag error 
  
  #MISSING DATA AND CLUSTERING:
  
  #Random forests deal with two types of missing data:
  #1) Missing data in the original dataset used to create the random forest
  #2) Missing data in a new sample that you want to categorize
  
  #For (1), the goal is to impute by considering similar samples:
  #(i) Start by imputing using unweighted averages calculated from whole dataset
  #(ii) Build random forest
  #(iii) Run all the data down all the trees; samples that end at the same leaf node are "similar"
  #(iv) Keep track of similarities from all trees in a proximity matrix with nrows=ncols=# of samples.
  #(v) Divide proximity vlaues by number of trees
  #(vi) Use proximity matrix values to calculate the weighted average/frequency for a missing value
  #     where the weight is calcluted as the share of 
  
  #Note: Proxmity matrix is the residual of distance matrix, meaning it can be used to draw heat maps or MDS plots
  
  #For (2):
  #(i) For new sample with missing dat, create two samples, one with classificaiton value TRUE and one for FALSE. 
  #(ii) Use iterative process for dealing with missing original data (see above) to get good estimates for missing data for both samples
  #(iii) Run the two samples down all the trees, see which is correctly labeled by the random forest the most times
  #(iv) Select sample that is correctly labeled by the forest the most times.
  
  
  #EXAMPLE:
  
  #Libraries
  library(ggplot2)
  library(cowplot)
  library(randomForest)
  
  #Get and format data
  {
    ## NOTE: The data used in this demo comes from the UCI machine learning
    ## repository.
    ## http://archive.ics.uci.edu/ml/index.php
    ## Specifically, this is the heart disease data set.
    ## http://archive.ics.uci.edu/ml/datasets/Heart+Disease
    
    url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    data <- read.csv(url, header=FALSE)
    
    #####################################
    ##
    ## Reformat the data so that it is 
    ## 1) Easy to use (add nice column names)
    ## 2) Interpreted correctly by randomForest..
    ##
    #####################################
    head(data) # you see data, but no column names
    
    colnames(data) <- c(
      "age",
      "sex",# 0 = female, 1 = male
      "cp", # chest pain 
      # 1 = typical angina, 
      # 2 = atypical angina, 
      # 3 = non-anginal pain, 
      # 4 = asymptomatic
      "trestbps", # resting blood pressure (in mm Hg)
      "chol", # serum cholestoral in mg/dl
      "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
      "restecg", # resting electrocardiographic results
      # 1 = normal
      # 2 = having ST-T wave abnormality
      # 3 = showing probable or definite left ventricular hypertrophy
      "thalach", # maximum heart rate achieved
      "exang",   # exercise induced angina, 1 = yes, 0 = no
      "oldpeak", # ST depression induced by exercise relative to rest
      "slope", # the slope of the peak exercise ST segment 
      # 1 = upsloping 
      # 2 = flat 
      # 3 = downsloping 
      "ca", # number of major vessels (0-3) colored by fluoroscopy
      "thal", # this is short of thalium heart scan
      # 3 = normal (no cold spots)
      # 6 = fixed defect (cold spots during rest and exercise)
      # 7 = reversible defect (when cold spots only appear during exercise)
      "hd" # (the predicted attribute) - diagnosis of heart disease 
      # 0 if less than or equal to 50% diameter narrowing
      # 1 if greater than 50% diameter narrowing
    )
    
    head(data) # now we have data and column names
    
    str(data) # this shows that we need to tell R which columns contain factors
    # it also shows us that there are some missing values. There are "?"s
    # in the dataset.
    
    ## First, replace "?"s with NAs.
    data[data == "?"] <- NA
    
    ## Now add factors for variables that are factors and clean up the factors
    ## that had missing data...
    data[data$sex == 0,]$sex <- "F"
    data[data$sex == 1,]$sex <- "M"
    data$sex <- as.factor(data$sex)
    
    data$cp <- as.factor(data$cp)
    data$fbs <- as.factor(data$fbs)
    data$restecg <- as.factor(data$restecg)
    data$exang <- as.factor(data$exang)
    data$slope <- as.factor(data$slope)
    
    data$ca <- as.integer(data$ca) # since this column had "?"s in it (which
    # we have since converted to NAs) R thinks that
    # the levels for the factor are strings, but
    # we know they are integers, so we'll first
    # convert the strings to integiers...
    data$ca <- as.factor(data$ca)  # ...then convert the integers to factor levels
    
    data$thal <- as.integer(data$thal) # "thal" also had "?"s in it.
    data$thal <- as.factor(data$thal)
    
    ## This next line replaces 0 and 1 with "Healthy" and "Unhealthy"
    data$hd <- ifelse(test=data$hd == 0, yes="Healthy", no="Unhealthy")
    data$hd <- as.factor(data$hd) # Now convert to a factor
    
    str(data) ## this shows that the correct columns are factors and we've replaced
    ## "?"s with NAs because "?" no longer appears in the list of factors
    ## for "ca" and "thal"
    
    
    #####################################
    ##
    ## Now we are ready to build a random forest.
    ##
    #####################################
  }
  
  set.seed(42)
  
  data.imputed<-rfImpute(hd ~ ., data=data, iter=6) #impute missing values in predictor data
    #iter=6 => # of random forests to build to estimate missing values
    #after each iteration,rfImpute() prints out the out-of-bag (OOB) error rate. This should get smaller
    #if the estimates are improving. Since they are not, we conclude that our estimates are as good as they are going to get
  
  model<-randomForest(hd ~ ., dat=data.imputed, proximity=TRUE) #build actual random forest
  model #sumamrize model (doesn't work with summarize() function)
    #Type of random forest=classificaiton. This is because the "dependent variable" is categorical. 
    #If dependent variable was continuous, it would say "regression"
    #Number of trees - number of decision trees in random forest. Default is 500
    #Number of random variables selected at each step = 3. This is default (square root of # of variables). Can be optimized 
    #OOB estimate: 17.16% of out-of-bag sample were incorrectly classified
    #Confusion matrix: gives specifics of error
  
  #Now check to see if the random forest is actually big enough.. you've made enough when the OOB no longer improves.
  oob.error.data <- data.frame(
    Trees=rep(1:nrow(model$err.rate), times=3),
    Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(model$err.rate)),
    Error=c(model$err.rate[,"OOB"], 
            model$err.rate[,"Healthy"], 
            model$err.rate[,"Unhealthy"]))
  
  #Plot convergence of error rates
  ggplot(data=oob.error.data, aes(x=Trees, y=Error)) + 
    geom_line(aes(color=Type))
  
  #Try random forest with more trees to see if error rates decline further
  model<-randomForest(hd ~ ., dat=data.imputed, ntree=1000, proximity=TRUE) #build actual random forest
  model #Out of bag error rate is same as before, meaning that adding more trees does not help
  
  #Now determine optimal number of variables to select at each split
  oob.values<-vector(length=10)#create empty vector
  for(i in 1:10){
    temp.model<-randomForest(hd ~ ., data=data.imputed, mtry=i, ntree=1000) #build model with # of vars=i
    oob.values[i]<-temp.model$err.rate[nrow(temp.model$err.rate),1] #collect OOB error rate
  }
  oob.values #3rd value corresponding to mtry=3 is optimal because of lowest OOB error rate
  
  #Draw MSD plot with samples
  distance.matrix<-dist(1-model$proximity) #distance matrix=1-proximity matrix
  
  mds.stuff<-cmdscale(distance.matrix, eig = TRUE, x.ret=TRUE) #'cmd'=classical multidimensional scaling
  mds.var.per<-round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)#calculate percentage of variation in the distance matrix that X and Y axes account for
  
  mds.values<-mds.stuff$points #formate data for ggplot
  mds.data<-data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=data.imputed$hd)
  
  ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) +  #draw graph
    geom_text(aes(color=Status)) +
    theme_bw() +
    xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
    ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
    ggtitle("MDS plot using (1 - Random Forest Proximities)") 
  
  rm(list = ls())
  
  
}

#AdaBoost
{
  #Three ideas behind AdaBoost:
  #1) AdaBoost combines a lot of "weak learners" (stumps, or trees with only a root and two leaves) to make classifications
  #2) Some stumps get more say in the classificaiton than others
  #3) Each stump is made by taking the previous stump's mistakes into account
  
  #Creating a forest of stumps with AdaBoost:
  #1) Give each sample a weight that indicates how important it is to be correctly classified. At the start, each sample gets the same weight: 1/N
  #2) Build first stump: find variable that does the best job classifying samples via GINI impurity
  #3) With results of first stump, assign priority to samples which are incorrectly classified.
  #   For samples that are incorrectly classified, new sample weight=sample weight*exp(amount of say)
  #   where 'amount of say'=0.5log((1-total error)/total error) and 'total error' is the sum of the weights
  #   associated with the incorrectly classified samples.
  #   For samples that are correctly classified, new sample weight=sample weight*-exp(amount of say)
  #4) Normalize new sample weights so they sum to 1
  #5) Bootstrap new data based on new weights (by sampling from weight distribution and duplicating observations)
  #6) Return to step 2 with the new data generated in step 5
  #7) Use polling of stumps weighted by amount of say to classify data.

  #EXAMPLE
  library(caret)
  library(dplyr)
  
  #format data:
  data("Titanic")
  Titanic=as.data.frame(Titanic) %>% select(!Freq)
  Titanic$Class<-as.character(Titanic$Class)
  Titanic$Class[Titanic$Class=="Crew"]<-"4th"
  Titanic$Class<-as.factor(substr(Titanic$Class, 1,1))
  
  #Data splitting
  index =  createDataPartition(Titanic$Survived, p=0.75, list=FALSE)
    #p=percentage of data that goes to training
  
  #Create training and testing data
  train<-Titanic[index,]
  test<-Titanic[-index,]
  
  #model 1
  model<-train(Survived~., method="adaboost", data=train, metric="Accuracy")

}


