# Project Details<br>
**Project title :** Hate Speech Detection<br>
**Topic** : Natural Language Processing - NLP <br>
**Group Members :**<br>
Tadikamalla Gowtham Krishna - 121321909<br>
Raahul Narayana Reddy Kummitha - 121109521<br>
Sriyank Sagi - 121302335<br>
Dhanush Garikapati - 121324924<br>
Venakata SatySai Maruti Kameshwar Modali - 121306050<br>
# Final Project: Hate Speech Detection


## Project Overview
- **Objective**: Develop a machine learning model to detect hate speech in text data, focusing on accurate classification into hate and non-hate categories.

 
This project focuses on Hate Speech Detection using Natural Language Processing (NLP) techniques and machine learning models to identify hateful text on social media. Motivated by the increasing prevalence of hate speech, the dataset reflects social media trends, including emoticons and slang, which complicate detection. It is preprocessed and categorized into hateful ("1") or non-hateful ("0") text, enabling the training of effective models. This benchmark dataset supports Deep Learning (DL) and NLP applications, aiding in the development of automated systems to filter harmful content while adhering to policy guidelines to reduce cyber harm.

---

## Key Steps in the Project
The dataset used for this project is a collection of social media comments sourced from open-access platforms like Kaggle. It contains text labeled as either hateful ("1") or non-hateful ("0"). The hateful category includes offensive language, while the non-hateful category contains neutral comments.

1.**Data preparation** involved several steps:

Preprocessing: Duplicates and null values were removed, and text was cleaned by removing special characters and numbers. Only English comments were retained using the langdetect library. The labels were verified for accuracy.
Transformation: The cleaned data was stored in a pandas DataFrame with two columns: "Text" and "Label." Text was tokenized and padded to a uniform length of 100 tokens.
Storage: The processed dataset was stored as a CSV file, enabling efficient querying and machine learning integration.
These steps were essential to ensure the dataset was clean, consistent, and ready for analysis, which is vital for building accurate machine learning models.

2. Exploratory Data Analysis (EDA) was conducted to understand the dataset's structure and key features. The following findings were made:

- **Label Distribution**: The dataset, containing hateful ("1") and non-hateful ("0") comments, was imbalanced, with a higher number of non-hateful comments. This imbalance was considered for machine learning adjustments.

- **Statistical Tests and Hypothesis Testing**:
  - **Hypothesis Testing**: A test was performed to check for a significant difference between the frequency of hateful and non-hateful comments. The null hypothesis, which stated there was no significant difference, was rejected.
  - **Chi-Square Test**: This test analyzed the relationship between specific words and their association with the labels. The results confirmed certain words' statistical significance in identifying hate speech.

- **Distribution Analysis**: The length of comments was examined, revealing that hateful comments were typically shorter than non-hateful ones. A box plot visualized this difference in comment lengths across the two classes.

3. **Machine Learning**:
   - Model: Support Vector Machine (SVM) was implemented for classification due to its performance on high-dimensional text data.
   - Evaluation Metrics: Accuracy, precision, recall, and F1-score were used to evaluate model performance.
   - Results:
     - Accuracy: 92%
     - F1-Score: 0.91

4. **Visualization**:
   - Example Code:
     ```python
     from wordcloud import WordCloud
     import matplotlib.pyplot as plt

     wordcloud = WordCloud(width=800, height=400).generate(" ".join(hate_speech_text))
     plt.figure(figsize=(10, 5))
     plt.imshow(wordcloud, interpolation='bilinear')
     plt.axis('off')
     plt.show()
     ```
   - Example Plot:
     ![Word Cloud](path/to/wordcloud.png)

---

## Repository Contents
- **Code Files**: 
  - `data_preprocessing.py`: Scripts for cleaning and preprocessing text data.
  - `eda_visualizations.ipynb`: Jupyter Notebook containing EDA and insights.
  - `ml_model_training.py`: Script for training and evaluating the SVM model.
- **Additional Files**: 
  - `requirements.txt`: Contains all dependencies required to execute the project.

---

## Visualizations
- Example Visualization:
  ![Visualization](path/to/visualization.png)
  - **Description**: This visualization highlights the most frequent words appearing in hate speech messages, providing insight into common linguistic patterns.

---



## Insights and Future Work
- **Key Insights**:
  - SVM performed effectively, achieving high accuracy and F1-score.
  - Certain terms were strong indicators of hate speech, which were visualized in the EDA phase.
- **Next Steps**:
  - Implement advanced NLP techniques such as transformers for improved performance.
  - Extend the dataset to include more diverse sources and languages.

---

## License
[Include license information.]
