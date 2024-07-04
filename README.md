# Exploratory Data Analysis for HousePrices
Here we will perform the Exploratory Data Analysis for the HousePrices Dataset from Kaggle.

Let's import the necessary Python libraries.

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    %matplotlib inline
    pd.pandas.set_option('display.max_columns',None)

Now we will read the dataset and visualise the top 5 rows to get some basic idea on the dataset.
    
    dataset = pd.read_csv('train.csv')
    dataset.head()

Now we have to find, if there were any missing values concerning the feature. If there were any missing values in them, we have to find the percentage of the missing values in that specific feature.
    
    features_with_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1]
    for feature in features_with_nan:
      print(feature,':',np.round(dataset[feature].isnull().mean(),4),'% of missing values')

Also, we need to find the number of missing values for each feature.
    
    for feature in features_with_nan:
      print(feature,':',dataset[feature].isnull().sum(),'missing values')

Here from the above two blocks of code, we can observe missing values in the data set. Now it is necessary that we need to see if there was any dependency of the missing values with the output, in this case (SalePrice).

We see the above observations by plotting the bar graphs.
    
    for feature in features_with_nan:
      data=dataset.copy()
      data[feature]=np.where(data[feature].isnull(),1,0)
      data.groupby(feature)['SalePrice'].median().plot.bar()
      plt.xlabel(feature)
      plt.ylabel("SalePrice")
      plt.title(feature)
      plt.show()

From the above graphs we observe that in some graphs, the 'one' bar is bigger than the 'zero' bar. In that case, then there must be an effect of those missing values on the output 'SalePrice'.

Upto here one step was completed and now it turns to check for the numerical variables and their dependency on the output.
    
    numerical_features = [feature for feature in dataset.columns if dataset[feature].dtype != 'O']
    print(numerical_features)
    print("Total no.of numerical features are ", len(numerical_features))

Visualise and understand the dataset of numerical features.

    dataset[numerical_features].head()

Here from the above dataset, we have data-time variables that are year features. These types of features are dependent on the past data. Now we will find the list of features that contain the year feature.
    
    year_feature=[feature for feature in dataset.columns if 'Yr' in feature or 'Year' in feature]
    print(year_feature)

Now we will see the unique year values corresponding to the year features.
    
    for feature in year_feature:
      print(feature, dataset[feature].unique())

Now we consider one parameter and we compare\check the dependency on the output SalePrice by plotting a graph.
    
    dataset.groupby('YrSold')['SalePrice'].median().plot()
    plt.xlabel('YrSold')
    plt.ylabel("SalePrice")
    plt.title("YrSold")
    plt.show()

Here from the above graph, we can observe that, as the year increases then the sales price gradually decreases.


Similar to the above case we will compare and observe for all remaining year features.
    
    for feature in year_feature:
      if feature != 'YrSold':
        data=dataset.copy()
    
        data[feature]=data['YrSold']-data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()

Here we have two types of numerical features in the above data sets.
  - continuous variables
  - discrete variables
  Now we will find the discrete variables of considering length less than 25 and they should not present in the year_features and 'Id'.
        
        discrete_features=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature + ['Id']]
        print(discrete_features)
        print(len(discrete_features))

Now let us find the relation between the discrete features and the output sale price but plotting some graphs.
    
    for feature in discrete_features:
      data=dataset.copy()
    
      data.groupby(feature)['SalePrice'].median().plot.bar()
      plt.xlabel(feature)
      plt.ylabel('SalePrice')
      plt.title(feature)
      plt.show()

Now check the data as per the plots plotted.
    
    for feature in discrete_features:
      print(feature,dataset[feature].unique(),len(dataset[feature].unique()))

Now we will see the categorical features that contain the sting as the value.
    
    continuous_features=[feature for feature in numerical_features if feature not in discrete_features + year_feature+['Id']]
    print(continuous_features)
    print(len(continuous_features))

Now we will check the relation between the continuous features and the sale price by plotting some histograms.
    
    for feature in continuous_features:
      data=dataset.copy()
      data[feature].hist(bins=25)
      plt.xlabel(feature)
      plt.ylabel('count')
      plt.title(feature)
      plt.show()

Now we will apply the logarithmic transformation to the continuous feature values.
    
    for feature in continuous_features:
      data=dataset.copy()
      if 0 in dataset[feature].unique():
        pass
      else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('sale price')
        plt.title(feature)
        plt.show()

Now we will find the outliers, as these outliers have some impact on the output. Outliers are the values of the features that are extremely maximum or extremely minimum. We can find if there were any outliers in the feature by plotting the box plot.
    
    for feature in continuous_features:
      data=dataset.copy()
      if 0 in data[feature].unique():
        pass
      else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()

From the above figures some values are present at extreme positions, ie they are outliers.

Up to now, the numerical features were completed, now it's turn for the categorical features to take into action. Now we find the categorical features.
    
    categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']
    dataset[categorical_features].head()

Now we will observe the relation between the categorical features and the saleprice.
    
    for feature in categorical_features:
      data=dataset.copy()
      data.groupby(feature)['SalePrice'].median().plot.bar()
      plt.xlabel(feature)
      plt.ylabel("SalePrice")
      plt.title(feature)
      plt.show()

Here the Exploratory Data Analysis part was completed. Next, we will go through with the Feature Engineering.

