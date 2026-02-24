# Lucas Systems Warehouse Planning Capstone Project

## Overview
This project develops statistical and machine learning models to estimate the expected completion time of warehouse work queues using large-scale execution data collected from real warehouses. The goal is to support proactive workforce planning by predicting how long a queue of tasks will take before work begins. The data comes from multiple warehouses with different layouts, products, and workflows, but all operate under the same voice-directed system (Jennifer), allowing the models to generalize across environments.

---

## Goals
- Estimate total completion time of a warehouse work queue in advance  
- Build models that generalize across warehouse layouts and customers  
- Identify operational factors that most strongly affect task duration  
- Provide interpretable insights for managers and planners  

---

## Data Sources

**Activity Data (`xx_Activity`)**
- Timestamped task completion records
- User ID, task type, product, and location
- Used to compute task duration via consecutive timestamps

**Location Data (`xx_Locations`)**
- Maps LocationID → Aisle, Bay, Level, Slot
- Enables spatial movement modeling

**Product Data (`xx_Products`)**
- Weight, volume, and dimensions
- Used to model handling difficulty

**Distance Matrices**
- Precomputed distances between warehouse locations
- Used to estimate travel distance between tasks

---

## Data Processing Pipeline
Key preprocessing steps:
- Sort tasks chronologically per worker
- Compute task duration from timestamp differences
- Remove assignment start events (`AssignmentOpen`)
- Filter unrealistic long gaps (breaks or interruptions)
- Join activity data with product and location tables
- Compute distance between consecutive tasks
- Engineer features such as log-transformed variables and categorical encodings

---

## Exploratory Analysis Highlights
- Task times are strongly right-skewed → log transformation improves modeling
- Travel distance shows a clear positive relationship with completion time
- WorkCode significantly affects time patterns
- Spatial factors (aisle, level) introduce systematic differences in duration

---

## Modeling Approaches

**Linear Regression**
- Baseline interpretable model
- Quantifies marginal effect of operational factors

**Decision Trees**
- Capture nonlinear relationships and interactions
- Provide feature importance rankings

**Regularized Regression (LASSO / ElasticNet)**
- Automatic feature selection
- Handles correlated predictors

**Neural Networks**
- Explored for flexibility
- Used mainly for comparison due to lower interpretability

---

## Technologies Used
- Python
- pandas / numpy
- scikit-learn
- statsmodels
- matplotlib / seaborn

---

## Impact
This project enables warehouse managers to:
- forecast workload completion time
- optimize staffing levels
- identify operational bottlenecks
- compare efficiency across warehouses


