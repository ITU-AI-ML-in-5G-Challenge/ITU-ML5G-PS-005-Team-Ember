# ITU-ML5G-PS-005-Team-Ember

Team Ember: Network anomaly detection based on logs 

## Authors

**Team Name:** Team Ember

**Team Members:**
- Longsheng Du
- Yu Du
- Ke Wu

## Files

[Report](Network%20Anomaly%20Detection%20Based%20on%20Logs.pdf)

[Slides](Network%20Anomaly%20Detection%20Based%20on%20Logs.pptx.pdf)

[Code](project)

## Requirments

Install all the packages:

```
numpy>=1.17.3
scikit_learn==0.23.1
pandas==1.1.5
torch==1.4.0
```

## Running the Codes

Use designated auto score code platform and set `app.py` as entry script. Or run `app.py` in proper environment with dataset.

Project is designed modularly, modules inside can be used outside the project.

```
SysIO:      Handle platform file input output.
LogParser:  Handle log parsing to templates.
LogFeature: Handle parsed templates and create sequence for model.
LogTrainer: Train the LSTM model using training sequence data.
LogTester:  Predict anomaly using LSTM model on test dataset.
```

# Credit

The code is inspired or based on these projects:

https://github.com/scikit-learn/scikit-learn  
https://github.com/pandas-dev/pandas  
https://github.com/numpy/numpy  
https://github.com/pytorch/pytorch  
https://github.com/logpai/logparser  
https://github.com/wuyifan18/DeepLog  
https://github.com/Thijsvanede/DeepLog  
https://github.com/Wapiti08/DeepLog  
