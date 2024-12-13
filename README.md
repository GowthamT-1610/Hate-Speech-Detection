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
- **Dataset**: Hate Speech Dataset (sourced from Kaggle).

---

## Key Steps in the Project
1. **Data Preparation**:
   - Text preprocessing steps included tokenization, removal of stop words, lemmatization, and lowercasing.
   - Addressed class imbalance using techniques such as [if applicable, specify oversampling or undersampling].

2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics:
     - Dataset contained [number] entries with [number] hate speech instances and [number] non-hate instances.
   - Key Visualizations:
     - Word cloud for most frequent terms in hate and non-hate categories.
     - Distribution of text length across categories.
   - Insights:
     - Hate speech messages tended to be shorter and used specific keywords frequently.

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

## Ethical Considerations
- Bias in training data was mitigated by balancing the dataset and ensuring diverse representation.
- Emphasized interpretability of the model to avoid misuse or misclassification.

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
