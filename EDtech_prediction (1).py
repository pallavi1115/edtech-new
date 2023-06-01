#import all required libraries 

import pandas as pd
#import dataset
data= pd.read_csv(r"C:\Users\Admin\Desktop\edtech prediction\edtech Data.csv")

#data preprocessing 

#EDA
data.info()
data.columns
data.dtypes

#checking for duplicates
duplicate = data.duplicated()
duplicate
sum(duplicate) 

#checking for null value
data.isna().sum().sum()

# import labelencoder for label encoding
from sklearn.preprocessing import LabelEncoder # Label Encoder
labelencoder = LabelEncoder()  # creating instance of labelencoer

#convert all categorical features in label encoding
data['name_of_institution']= labelencoder.fit_transform(data['name_of_institution'])
data['Level_of_the_institution']= labelencoder.fit_transform(data['Level_of_the_institution'])
data['name_of_course']= labelencoder.fit_transform(data['name_of_course'])
data['location']= labelencoder.fit_transform(data['location'])
data['mode_of_course']= labelencoder.fit_transform(data['mode_of_course'])
data['Pre_recorded_session']= labelencoder.fit_transform(data['Pre_recorded_session'])
data['certification']= labelencoder.fit_transform(data['certification'])
data['Qualification_of_instructor']= labelencoder.fit_transform(data['Qualification_of_instructor'])
data['maintenance']= labelencoder.fit_transform(data['maintenance'])
data['Marketing_cost']= labelencoder.fit_transform(data['Marketing_cost'])
data['Placement']= labelencoder.fit_transform(data['Placement'])
data['course_level']= labelencoder.fit_transform(data['course_level'])
data['mentor_for_doubt_clarification']= labelencoder.fit_transform(data['mentor_for_doubt_clarification'])
data['availability_Mentor_for_doubt_clarification']= labelencoder.fit_transform(data['availability_Mentor_for_doubt_clarification'])
data['Technology']= labelencoder.fit_transform(data['Technology'])
data['Course_content']= labelencoder.fit_transform(data['Course_content'])
data['Rating_of_course']= labelencoder.fit_transform(data['Rating_of_course'])

#data typess
data.dtypes

#function for removing k from two features
def convert_to_numeric(value):
    if value.endswith('k'):
        return int(value[:-1]) * 1000
    else:
        return int(value)

# Apply the conversion function to the 'Both' column
data['No_of_students'] = data['No_of_students'].apply(convert_to_numeric)
data['Instructor_salary'] = data['Instructor_salary'].apply(convert_to_numeric)

#seprate the independant and dependant features 
x=data.iloc[:,1:17] #Independant variable
y=data["Price"]  #Dependant variable

#............................................................................
#features selection method
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#............................................................................ 

#selecting features who has more feature importance
x1 = data[["Qualification_of_instructor","name_of_course" , "Duration" ,"course_level","Course_content","Marketing_cost","No_of_instructor","maintenance","No_of_students","Rating_of_course"]]
y1=data['Price']

#split data into train and test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=32)

### Linear Regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Fit the model on the training data
regressor.fit(x_train, y_train)

# Predict on the testing data
y_pred = regressor.predict(x_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
#mean squared method
from sklearn.metrics import mean_squared_error

#######################################################################

from sklearn.ensemble import RandomForestRegressor
# Create a Random Forest model
model = RandomForestRegressor()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the evaluation metric
print("Mean Squared Error:", mse)

from sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [1,2,3,4],
}
regr = RandomForestRegressor(random_state=0)

clf = GridSearchCV(regr, parameters)
clf.fit(x_train, y_train)

y_pred_train = clf.predict(x_train)
mean_squared_error(y_train, y_pred_train)

y_pred = clf.predict(x_test)
mean_squared_error(y_test, y_pred)

###################################################################
#decision tree regression 
# Import the required libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create an instance of the DecisionTreeRegressor class
regressor = DecisionTreeRegressor()

# Fit the model on the training data
regressor.fit(x_train, y_train)

# Predict on the testing data
y_pred = regressor.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

#create pickle file
import pickle
pickle.dump(model,open('model.pkl','wb'))

model=pickle.load(open("model.pkl",'rb'))
print(model.predict([[1,1,1,1,1,1,1,1,1,1]]))
 