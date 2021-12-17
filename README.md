# Tack Prediction

A marine electronics company that produces hardware and software for sailing yachts, installed some sensors on one of the boats and provided the dataset they have collected. 

We analyse the data in order to give the client some feedback on data collection and handling process, and suggest some ideas of how this data can be used to enhance their product and make it more popular among professional sailors and boat manufacturers. In particular, we are interested in 'tack prediction'. "A tack is a specific maneuver in sailing and alerting the sailor of the necessity to tack in the near future would bring some advantage to them compared to other sailors, who would have to keep an eye out on the conditions all the time to decide when to tack." We build a forecasting model that would be alerting sailors of the tacking event happening ahead.

features.txt contains the description of the data columns given in the dataset. requirements.txt contains a list of Python packages required by the codes. (Note: tensorflow and keras are commented out in the codes and not used.)

In exploratory_data_analysis.py, we perform exploratory data analysis (EDA) and generate plots which are stored in the 'plots' folder. The EDA also informs us of feature engineering strategy, e.g. convert some angles to sines and cosines, and some to the principal value (between -180 deg and 180 deg). 


For tack prediction, we aggregate our data by minute for two reasons. First, we only predict 'Tacking' but not other variables, and given current and past knowledge on them we can only predict one step ahead. Aggregating by minute allows the model to have more time (1 min VS 1 sec) to train and make predictions. Second, it smooths out the data. We use rolling window forecasting with an 18-hour (1080 minute) training window and a 1-minute prediction window. 


In rolling_forecast_feature_selection.py, using logistic regression, we explore feature selection and their lag features, and evaluate different choices by F_beta scores, the confusion matrix, the precision score and the recall score. It turns out that, for best performance, it is important to include most of the given features, except for DateTime, latitude, longitude and rudder angle ('RudderAng'). As for the lag features, the F-scores suggest the use of lags = 5 or 10. We also find that the precision score decreases with the number of lags included, while the opposite is true for the recall and accuracy scores. 


In rolling_forecast_model_selection.py, with the optimal features (and their lags), we find that among logistic regression and XGBoost Classifier, XGBoost performs best, in fact far better than logistic regression (F-scores of ~0.8 VS 0.5), despite the former takes longer to fit. 


All in all, the best validation scores are: F_1: 0.855, F_2: 0.878, F_0.5: 0.832, precision: 0.818 and recall: 0.894 coming from lags = 10; F_1: 0.833, F_2: 0.872, F_0.5: 0.798, precision: 0.775 and recall: 0.901 coming from lags = 5. XGBoost with lags = 10 takes 60% more time to fit and predict than that with lags = 5. Looking at the confusion matrices, the differences are minute. For deployment on a boat where the onboard computational power and Internet access is limited, I recommend using lags = 5.

Note that we advise that we not use lag features of the target label ('Tacking'), as this would be like having a human sailing instructor onboard telling us when to tack, which the machine takes as input to decide whether it should tell you also to tack. Indeed, we find that including such feature would make our machine learning model almost perfect. But it defeats the purpose: our goal is to build a model that tells us when to tack, without human intervention.

rolling_forecast_master.py is the final script to be used for model deployment.
