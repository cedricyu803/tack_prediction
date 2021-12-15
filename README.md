# Tack Prediction

The Client, a marine electronics company which produces hardware and software for sailing yachts, installed some sensors on one of the boats and provided the dataset they’ve collected. 

We analyse the data in order to give the client some feedback on data collection and handling process, and suggest some ideas of how this data can be used to enhance their product and make it more popular among professional sailors and boat manufacturers. In particular, we are interested in 'tack prediction'. "A tack is a specific maneuver in sailing and alerting the sailor of the necessity to tack in the near future would bring some advantage to them compared to other sailors, who would have to keep an eye out on the conditions all the time to decide when to tack." We build a forecasting model that would be alerting sailors of the tacking event happening ahead.

In exploratory_data_analysis.py, we perform exploratory data analysis and generate plots which are stored in the 'Plots' folder.

In fix_window_forecast.py, we perform data preprocessing, feature engineering and build fix-window forecast machine learning models for tack prediction. In rolling_forecast.py, we use rolling window forecasting with a 18 hour training window and 12 hour validation/prediction windows.
