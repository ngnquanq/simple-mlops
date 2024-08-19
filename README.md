Enhancing Customer Retention with Explainable AI (XAI) Insights
==============================
This project is my Undergraduate thesis at UEH. "Enhancing Customer Retention with Explainable AI (XAI) Insights" focuses on tackling the challenge of churn prediction. The primary goal is to develop and implement machine learning models that accurately predict which customers are most likely to churn and, more importantly, provide explainable insights into the reasons behind these predictions.

Churn prediction is a critical task for businesses as retaining existing customers is often more cost-effective than acquiring new ones. However, while many predictive models can identify potential churners, they often fail to explain why a customer is likely to leave. This lack of transparency can limit the effectiveness of intervention strategies, as decision-makers may not understand the key drivers behind customer churn.

To address this, the project leverages Explainable AI techniques such as SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations). These methods provide interpretable explanations for the predictions made by the churn models, helping businesses identify the specific factors—such as usage patterns, customer demographics, or engagement metrics—that contribute most significantly to churn risk.

By combining predictive power with explainability, this thesis aims to enhance customer retention efforts by enabling businesses to take targeted, informed actions based on clear insights into customer behavior. The project contributes to the field of customer relationship management (CRM) by demonstrating how Explainable AI can be applied to improve both the accuracy and the usability of churn prediction models, ultimately helping businesses reduce churn rates and improve customer satisfaction.

Key Findings and Insights for decision making 
==============================


Project Organization
==============================

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external                     <- Data from third party sources.
    │   ├── interim                      <- Intermediate data that has been transformed.
    │   ├── processed                    <- The final, canonical data sets for modeling.
    │   └── telecom_churn.csv            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

How to run code
==============================

## 1. Install requirement packages

Run this command in the CLI at the root of the directory:
```cmd
python -m pip install -r requirements.txt
```
## 2. Download the dataset
The process of downloading the dataset can be achieved via using command line or manually download the csv file on Kaggle. 
### 2.1 Via CLI. 
In order to be able for downloading the dataset via cli. One must have the API key to be export. We can navigate to the Kaggle page in take the key of it. The process of getting the API key was demonstrated clearly in [this document](https://www.kaggle.com/docs/api). After generate new token, a `kaggle.json` file will be downloaded, now you need to put it in the root directory. **Don't worry because the `.gitignore` will ignore your token won't be expose in case you fork and re-push it on your repository**.

A function had been created so that you will have to run the script like this to export the API:
```cmd
python export_kaggle_api.py
```
After that run the following command to download data directly to the folder `data` or elsewhere to suit the user's needed.

```cmd
kaggle datasets download kashnitsky/mlcourse -f telecom_churn.csv -p /data
```
After this process you will see a `telecom_churn.csv` file in the `simple-mlops\data` folder. 
### 2.2 Manually download the data 
The dataset was published for everyone on kaggle to use, please download the data directly from [this link](https://www.kaggle.com/datasets/kashnitsky/mlcourse?select=telecom_churn.csv) then extract it to the `data` folder. 

## 3. Run code
**Still working on this**
# Todo list:
- [ ] Code the parameter tuning part
- [ ] Comparing metrics 
- [ ] More dataset if neccessary
- [ ] Do the XAI part
- [ ] Find  **main insights** and propose ways for increasing business value
- [ ] Restructure repository structure
- [ ] Write reports
- [ ] Finish report
# Further developments
- Create an entire pipeline for this process.
# Contact information
Please reach me via my email: [tommyquanglowkey2011@gmail.com](tommyquanglowkey2011@gmail.com) or [quangnguyen.31211027664@st.ueh.edu.vn](quangnguyen.31211027664@st.ueh.edu.vn)
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

