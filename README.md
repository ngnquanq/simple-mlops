Enhancing Customer Retention with Explainable AI (XAI) Insights
==============================
This project is my Undergraduate thesis at UEH. "Enhancing Customer Retention with Explainable AI (XAI) Insights" focuses on tackling the challenge of churn prediction. The primary goal is to develop and implement machine learning models that accurately predict which customers are most likely to churn and, more importantly, provide explainable insights into the reasons behind these predictions.

Churn prediction is a critical task for businesses as retaining existing customers is often more cost-effective than acquiring new ones. However, while many predictive models can identify potential churners, they often fail to explain why a customer is likely to leave. This lack of transparency can limit the effectiveness of intervention strategies, as decision-makers may not understand the key drivers behind customer churn.

To address this, the project leverages Explainable AI techniques such as SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations). These methods provide interpretable explanations for the predictions made by the churn models, helping businesses identify the specific factors—such as usage patterns, customer demographics, or engagement metrics—that contribute most significantly to churn risk.

By combining predictive power with explainability, this thesis aims to enhance customer retention efforts by enabling businesses to take targeted, informed actions based on clear insights into customer behavior. The project contributes to the field of customer relationship management (CRM) by demonstrating how Explainable AI can be applied to improve both the accuracy and the usability of churn prediction models, ultimately helping businesses reduce churn rates and improve customer satisfaction.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

