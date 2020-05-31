import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks   import EarlyStopping

from imblearn.under_sampling import CondensedNearestNeighbour

df = pd.read_csv('frauddata.csv')


df.columns
df = df.rename(columns={'nameOrig':'senderid','oldbalanceOrg':'senderoldbal'
                        ,'newbalanceOrig':'sendernewbal','nameDest':'destid',
                        'oldbalanceDest':'destoldbal','newbalanceDest':'destnewbal'
                        })
df.info()
summary = df.describe()

df.dtypes # 3 string values.Only 'type' is categorical value

df.isnull().sum() #No NaN values.

sns.countplot(x='type',data=df)
#-----------------------------------------------------------------------

#VISUALIZATION to understand data columns

sns.boxplot(x=df['amount'])

sns.lineplot(x='step',y='amount',data=df)


#EDA--------------------------------------------------------------------

# 16 transfers where amount transacted was 0-> strange
zero_amount = df[df['amount']==0] # FRAUD

sns.countplot(x='isFraud',data=df) #highly imbalanced data

len(df[df['isFraud']==1])
#Total frauds=8213

8213/(df.shape[0]) # 0.12% of cases are fraud

# type of fraud transfers
df[df['isFraud']==1].groupby(by='type')['type'].value_counts()

# There are only 2 types of payment for all fraud cases- CASHOUT AND TRANSFER. 
#4116 CASHOUTS AND 4097 TRANSFERS

frauds = df[df.isFraud==1]

sns.countplot(x='type',data=frauds)
#For 4097 cases fraud is committed by first transferring out funds to # another account and subsequently cashing it out. 19 cases involve #direct cash out

notfrauds = df[df.isFraud==0]

# Checking if dest of transfers act as sender for cashouts which is #expected in a transfer first then cash out fraud 

fraudtransfer = frauds[frauds.type=='TRANSFER']
fraudcashout = frauds[frauds.type=='CASH_OUT']
fraudtransfer.destid.isin (fraudcashout.senderid).any()

# This is weird that dest of transfer is never a sender in cashout for frauds #no pattern here-> we can drop senderid and destid

#-----------------------------------------------------------------------

#eda on isflaggedfraud

flaggedfraud = df[df.isFlaggedFraud==1] 
sns.countplot(x='isFlaggedFraud',data=df)

# 16rows
#only transfer types
# we need to see if there are any patterns for flagging fraud

#is flagged fraud is said to be set according to kaggle's data description when amt>200,000.We need to check this
print("The number of times when isFlaggedfraud is not set,\
      despite the amount being transacted was >200,000 = {}".format(
      len(df[(df.isFlaggedFraud==0) & (df.amount>200000)])))
#This shows that isflaggedfraud is not a proper right variable


print("Maxm amount transacted when isflagged was not set ={}".format(
        max(df[(df.isFlaggedFraud==0) & (df.amount>200000)]['amount'])))
#92445516 not flagged fraud-> bad variable


#Moreover sender and recipient's old and new balance remained same. This is # because maybe these transactions were halted by banks.WE shall check if #there are cases where old and new balance is same, yet not flagged as fraud
df[(df.isFlaggedFraud==0) & (df['type']=='TRANSFER') & (df['destoldbal']== 0) & (df['destnewbal']== 0)]
#4158 rows-> Those are not flagged fraud-> This is not a definite pattern


#Are the people who have been flagged fraud transacted any other time
notflaggedfraud = df[df.isFlaggedFraud==0] 
flaggedfraud['senderid'].isin (pd.concat([notflaggedfraud['senderid'],notflaggedfraud['destid']])).any()
# False means no flagged fraud originator ever transacted. It was only one time he transacted


flaggedfraud['destid'].isin (pd.concat([notflaggedfraud['senderid'],notflaggedfraud['destid']]))
#True means a flagged recipient have transacted more than once
#index 2736446 and index 3247297


notflaggedfraud.iloc[2736446]
notflaggedfraud.iloc[3247297] 
# This person has conducted a genuine transaction as well-> impt data point


#since there are 16 rows of isflagged and no definite pattern can be found by our eda,we  will drop this column


#-----------------------------------------------------------------------
#eda on MERCHANTS


#NOW WE FIND INFO ABOUT rows containing sender or receiver name starting with #M.These are the merchants.we see the type of payment they are involved in  #and if they can be involved in any fraud
merchants =df[df.senderid.str.contains('M')] 
# merchants are never senders


merchants =df[df.destid.str.contains('M')] 
#merchants have always been recipients

merchants['type'].nunique()
merchants['type'].unique()
# all the merchants have been involved in payment type


merchants[merchants['isFraud']==1]
# empty dataframe means merchants are not involved in any fraud. We can safely drop these merchant rows
#--------------------------------------------------------------------------     
#some other eda


# check if there are cases where amount sent>senderoldbalance as this should not be possible
df[df.amount>df.senderoldbal]
##4 million rows have been incorrectly calculated


# check if there are cases where amount received>destnewdbalance as this should not be possible
df[df.amount>df.destnewbal]
# again 2.6 million incorrect calculations


#checking occurences where amt>0 but destold and new balance are both 0
df[(df['amount']>0) & (df['destoldbal']==0) &(df['destnewbal'] == 0) &
    (df['isFraud']==1)]
4070/len(frauds)
#50% of fraud transactions see old and new dest bal same. We cant impute these values


df[(df['amount']>0) & (df['destoldbal']==0) &(df['destnewbal'] == 0) &
    (df['isFraud']==0)]
2313206 /len(notfrauds) # 36% rows


#checking occurences where amt>0 but senderold and new balance are both 0
df[(df['amount']>0) & (df['senderoldbal']==0) &(df['sendernewbal'] == 0) &
    (df['isFraud']==1)]
25/len(frauds) #0.3% cases


df[(df['amount']>0) & (df['senderoldbal']==0) &(df['sendernewbal'] == 0) &
    (df['isFraud']==0)]
2088944/len(notfrauds)
#32% rows

# for fraud cases, dest balance remains 0 most of the times, but thats not the case for sender in a fraud cases.

#-----------------------------------------------------------------------


#making variables sendererror and desterror to determine error noticed in a transaction
df['sendererror'] = df['senderoldbal']+df['amount']-df['sendernewbal'] 
df['desterror'] = df['destoldbal']+df['amount']-df['destnewbal'] 
head = df.head(50)

#making variables from step- hours,day,dayofweek

num_hours = 24
frauds['hour'] = frauds.step % num_hours
notfrauds['hour'] = notfrauds.step % num_hours
df['hour'] = df.step % num_hours


list=[]
for i in range(0,len(frauds)):
    step = frauds.iloc[i].step
    list.append(np.ceil(step/num_hours))
frauds['day']=list

frauds['dayofweek'] = frauds['day']%7 

# result from 0->6 where 0 can be
#any day. if 0 was monday, 1 would be tuesday but if 0 is tue , 1 is wed


plt.hist(x='day',data=frauds)
# no definite pattern based off the day
plt.hist(x='hour',data=frauds)
# no definite pattern based off hour
plt.hist(x='dayofweek',data=frauds)
# no definite pattern based off dayofweek

#-----------------------------------------------------------------------

#visualization
sns.scatterplot(y='amount',x='step',data=df,hue='isFraud')

#heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(),annot=True)


#-----------------------------------------------------------------------
#DATA PREPROCESSING

df2 = df.copy()

#since frauds are only for type=cashout or transfer acc to analysis
X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
#SINCE TYPE=PAYMENT IS NOT INCLUDED ALL MERCHANTS HAVE #BEEN DROPPED


#Acc to analysis we dont consider flagged fraud for dependent variable
y = X['isFraud']


#DROP COLUMNS NOT NECESSARY
X = X.drop(['senderid', 'destid', 'isFlaggedFraud'], axis = 1)
X = X.drop('isFraud',axis=1)
 # remove dep variable from matrix of features


#check proportion of fraud data
sns.countplot(y,data=y) #imbalanced data
y[y==1] #8213 frauds
8213/len(y) # 0.3% fraud cases

#_----------------------------------------------------------------------

#HANDLING MISSING DATA

Xfraud = X.loc[y == 1] #All fraud cases
XnonFraud = X.loc[y == 0] # all non fraud cases


X[(X['amount']!=0) & (X['destoldbal']==0) &(X['destnewbal'] == 0)]
X[(X['amount']!=0) & (X['senderoldbal']==0) & (X['sendernewbal'] == 0)]
1308566/len(X)
#Around 47% of senders have 0 before and after values. Since 47% is a big value we wont impute it.Rather we can replace 0 by -1 #which will give a clear distinction and also it will be good for our model


#lets see how many are fraud in these 47%rows

index = X[(X['amount']!=0) & (X['senderoldbal']==0) & (X['sendernewbal'] == 0) ].index
li=[]
li.append(y.loc[index])
li[0].value_counts()
# only 25 cases are fraud, maxm cases are genuine for above pattern


X.loc[(X.senderoldbal == 0) & (X.sendernewbal == 0) & (X.amount != 0), \
      ['senderoldbal', 'sendernewbal']] = - 1
      
X[X['senderoldbal']==-1]
X[X['sendernewbal']==-1].head()

#-----------------------------------------------------------------------

#Categorical Values(ENCODING)


X.dtypes #Type datatype is object
X.loc[X.type == 'TRANSFER','type' ] = 0 #TRANSFER =0
X.loc[X.type == 'CASH_OUT', 'type'] = 1  # CASHOUT =1
X.dtypes #Type datatype is int

#----------------------------------------------------------------------    

#TRAIN TEST SPLITTING
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#-----------------------------------------------------------------------

#CLASSIFICATION TECHNIQUES

#1)LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
lclassifier = LogisticRegression(random_state = 0)
lclassifier.fit(X_train, y_train)
y_predl = lclassifier.predict(X_test)

from sklearn.metrics import confusion_matrix,r2_score,classification_report
cmL = confusion_matrix(y_test, y_predl)
cmL
r2_score(y_test,y_predl) #46%
report = classification_report(y_test,y_predl)
# as expected logistic regression works bad on imbalanced data
#-------------------------------------------------------------------------

#2)Random Forest
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfclassifier.fit(X_train, y_train)
y_predrf = rfclassifier.predict(X_test)
rfcm = confusion_matrix(y_test, y_predrf)
rfcm
r2_score(y_test,y_predrf) # 81%
report = classification_report(y_test,y_predrf)


# changing the number of estimators
rfclassifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
rfclassifier.fit(X_train, y_train)
y_predrf = rfclassifier.predict(X_test)
rfcm = confusion_matrix(y_test, y_predrf)
rfcm
r2_score(y_test,y_predrf) # 82%
report = classification_report(y_test,y_predrf)
#precision as well as recall both are good

#-----------------------------------------------------------------------
#implement xgbosot
from xgboost import XGBClassifier
xgbclassifier = XGBClassifier()
xgbclassifier.fit(X_train, y_train)
y_predxgb = xgbclassifier.predict(X_test)
cmxgb = confusion_matrix(y_test, y_predxgb)
cmxgb
r2_score(y_test,y_predxgb) # 87%
reportxgb = classification_report(y_test,y_predxgb)
#-----------------------------------------------------------------------

#ANN

#converting dep variable in array for neural nets

y_train = y_train.values
y_test = y_test.values

model = Sequential()
#1st hidden layer
model.add(Dense(units=5,input_dim=10,activation='relu'))

model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=40,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          
          validation_data=(X_test, y_test),
          callbacks=[early_stop] 
          )

#scale back y_predann
print(r2_score(y_test,y_predann)) #71%
# Mediocre performance

#-----------------------------------------------------------------------

#evaluating kfold and rforest by stratified k fold cross validation

accuracyrf=[]
accuracyxgb=[]
#empty list to store accuracies of all folds

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X,y)
for train_index, test_index in skf.split(X, y):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train_strat, X_test_strat = X.iloc[train_index], X.iloc[test_index]
  y_train_strat, y_test_strat = y.iloc[train_index], y.iloc[test_index]

  from sklearn.metrics import accuracy_score
  #evaluating rf classifier on 10 folds
  rfclassifier.fit(X_train_strat,y_train_strat)
  y_predrfstrat = rfclassifier.predict(X_test_strat)
  score = accuracy_score(y_test_strat,y_predrfstrat)
  accuracyrf.append(score)
np.array(accuracyrf).mean()
# Drop in accuracy for last fold. It means that the model must have started overfitting at some point


"""Feature importance"""
importance = rfclassifier.feature_importances_
feature_importance = pd.DataFrame(importance,columns=['importance'])
feature_importance = feature_importance.sort_values(by=['importance'],ascending=False)
colname=[]
for i in feature_importance.index:
    colname.append(X_train_strat.columns[i])
feature_importance['colname']=colname
#step variable has maxm contribution towards results


#evaluating xgboost classifier on 5 folds
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X,y)
for train_index, test_index in skf.split(X, y):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train_strat, X_test_strat = X.iloc[train_index], X.iloc[test_index]
  y_train_strat, y_test_strat = y.iloc[train_index], y.iloc[test_index]
  xgbclassifier.fit(X_train_strat,y_train_strat)
  y_predxgbstrat = xgbclassifier.predict(X_test_strat)
  score = accuracy_score(y_test_strat,y_predxgbstrat)
  accuracyxgb.append(score)
np.array(accuracyxgb).mean()
#97% accuracy after all the folds-> really good accuracy, No overfitting


#-----------------------------------------------------------------------

#xgboost and random forest performed well as expected for imbalanced
#data. Lets try to balance it and then try again


""" TAKES LOT OF TIME
#Performing CONDENSED NEAREST NEIGBOURING(CNN) undersampling technique
undersample = CondensedNearestNeighbour(n_neighbors=1)
#undersampling only the train set and not test set since doing on both
# the model may perform well, but will do bad on new data which comesimbalanced
X1_train, y1_train = undersample.fit_resample(X_train, y_train)
"""


# we need to resort to random under sub sampling
X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train)


fraudlen = len(y_train_df[y_train_df==1].dropna()) #5748 frauds in train set

#fetching indices of frauds
fraudindices = y_train_df==1
fraudindices = fraudindices[fraudindices==1]
fraudindices = fraudindices.dropna().index.values


# fetching indices of genuine transactions
genuineindices = y_train_df==0
genuineindices = genuineindices[genuineindices]
genuineindices = genuineindices.dropna().index.values


#randomly select indices from majority class
rand_genuineindex = np.random.choice(genuineindices, fraudlen, replace = False)
#concatenate fraud and random genuine index to make final training df
index = np.concatenate([fraudindices,rand_genuineindex])


#sampling to be done only on train set
X_train_balanced = X_train_df.iloc[index]
y_train_balanced = y_train_df.iloc[index]


sns.countplot(0,data=y_train_balanced) # balanced data
#-----------------------------------------------------------------------
#applying models now

#1)LOGISTIC REGRESSION
lclassifier.fit(X_train_balanced, y_train_balanced)
#prediction to be done on imbalanced test set
y_predl = lclassifier.predict(X_test)
cmL = confusion_matrix(y_test, y_predl)
cmL
r2_score(y_test,y_predl) #46%
report = classification_report(y_test,y_predl)


#-------------------------------------------------------------------------
#2)Random Forest
rfclassifier1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfclassifier1.fit(X_train_balanced, y_train_balanced)
y_predrf = rfclassifier1.predict(X_test)
rfcm = confusion_matrix(y_test, y_predrf)
rfcm
r2_score(y_test,y_predrf) 
report = classification_report(y_test,y_predrf)
# Recall increased as we wanted but precision dropped badly


# changing the number of estimators
rfclassifier2 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
rfclassifier2.fit(X_train_balanced, y_train_balanced)
y_predrf = rfclassifier2.predict(X_test)
rfcm = confusion_matrix(y_test, y_predrf)
rfcm
r2_score(y_test,y_predrf) 
report = classification_report(y_test,y_predrf)
#No improvement
#-----------------------------------------------------------------------


#3) implement xgboost
xgbclassifier.fit(X_train_balanced.values, y_train_balanced.values)
y_predxgb = xgbclassifier.predict(X_test)
cmxgb = confusion_matrix(y_test, y_predxgb)
cmxgb
r2_score(y_test,y_predxgb) 
report = classification_report(y_test,y_predxgb)
#no improvement
#-----------------------------------------------------------------------


#After undersampling randomly, the model performed worse. Although it
#managed to increase recall, precision and f1 score worsened. This is
#highly possible because of 2 reasons:
#A) We performed random sampling and couldnt do CNN undersampling as it takes lot of time in transforming such huge data
#B) our test set is not balanced(since we want practical real life test set)), so it is expected model performs bad 

# Results- The best performance was given for imbalanced dataset by XGBoost model. Although random forest gave similar
# r2 score on imbalanced data(little less), but the model stability was much greaeter in xgboost and hence is expected to work better on new data