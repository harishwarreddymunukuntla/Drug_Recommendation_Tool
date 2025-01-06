# Drug Recommendation Based on Patient Reviews Using Machine learning
## Overview
This project focuses on analyzing sentiment in drug reviews to provide insights and recommendations for pharmaceutical products. Using machine learning and deep learning techniques, the project predicts sentiments from user reviews and ranks drugs based on positive reviews and their usefulness.
## Dataset
The Drugs Review datasets (Drugs.com) used in this study was extracted from UCI Machine learning repository. These datasets do not contain personally identifiable information (PII) and they were available publicly and This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. The dataset consists of patient feedback on various medications, along with the associated medical. 
The data is divided into two parts: 75% for training and 25% for testing.

The datasets contain 7 features:
| Feature |	Type	| Description |
| :-----:| :-----:| :----------: |
| Id | Integer	| Unique Id for each entry |
| DrugName |	text	| Name of the drug |
| Condition |	text	| Health condition such as Headache, Pain, Depression etc. |
| review	| text	| Customer review about the drug |
| rating |	Integer	| Rating from 1 to 10 |
| date	| dd-mm-yy(date) |	Date when the review was submitted |
| UsefulCount	| Integer	| The no of people thought the review was useful |

### Dataset Link: 
https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com
## Installation

Clone the repository: 
   ```bash
   git clone https://github.com/harishwarreddymunukuntla/Drug_Recommendation_Tool.git
```
or

download the code zip file from https://github.com/harishwarreddymunukuntla/Drug_Recommendation_Tool.git and extract it.

Now run jupyter notebook and open DRTFinal.ipynb file. Change the dataset path accordingly. I converted dataset files from .tsv file to .csv file but you can use .tsv files itself. 

- train_df = pd.read_csv(r"<"folderpath">\drugLibTest_raw.tsv", sep='\t')
- test_df = pd.read_csv(r"<"folderpath">\drugLibTest_raw.tsv", sep='\t')

## Project Steps:

#### 1. Import Libraries
Load essential libraries such as pandas, scikit-learn, TensorFlow, matplotlib, etc.
#### 2. Load and Preprocess the Dataset
- Read the dataset. 
- Handle missing values in key columns (review, rating, etc.). 
- Remove special characters from text reviews. 

#### 3. Exploratory Data Analysis
#### 4. Sentiment Labeling
- Create sentiment labels:
- Binary Sentiment: Positive (rating ≥ 5) or Negative (rating < 5).
- Multi-Class Sentiment: Negative (rating < 5), Neutral (rating 5-6), Positive (rating > 6).
#### 5. Data Preparation
- Split data into training and validation sets.
- Apply text preprocessing techniques like TF-IDF transformation or tokenization.
#### 6. Model Development
##### a. LinearSVC
- Train a Linear Support Vector Classifier using TF-IDF features.
- Evaluate performance with accuracy, classification reports, and confusion matrices.
##### b. Logistic Regression
- Train a Logistic Regression model using TF-IDF features.
- Evaluate performance and visualize confusion matrices.
##### c. Convolutional Neural Network (CNN)
- Preprocess text into sequences with padding.
- Build and train a CNN with embedding and convolutional layers.
- Evaluate the model on validation data.
#### 7. Visualization of Training Performance
Plot training vs. validation accuracy and loss for CNN.
#### 8. Drug Sentiment Analysis
- Analyze the sentiment of reviews for specific conditions (e.g., "Headache").
- Aggregate positive review counts and useful counts per drug.
#### 9. Generate Top Recommendations
- Rank drugs based on sentiment analysis results for each model (LinearSVC, Logistic Regression, and CNN).
- Display the top 5 recommended drugs.

