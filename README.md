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

**Exploratory Data Analysis (EDA)**


Exploratory Data Analysis (EDA) was conducted to gain insights into the dataset, including its structure, distribution, and key characteristics.

Label Distribution: The dataset consists of comments labeled as either hateful ("1") or non-hateful ("0"). The EDA revealed an imbalance, with more non-hateful comments than hateful ones. This imbalance was addressed during the machine learning process.

**Statistical Tests and Hypothesis Testing**:

Hypothesis testing was performed to examine if there was a significant difference in the frequency of hateful and non-hateful comments. The null hypothesis was rejected based on the results.
A Chi-Square test was performed to analyze the relationship between specific words and their association with hateful or non-hateful labels, confirming the statistical significance of certain terms in identifying hate speech.
Distribution Analysis: The length of comments (in terms of word count) was analyzed, showing that hateful comments tend to be shorter than non-hateful ones. A box plot was created to visualize this distribution, highlighting the difference in comment lengths between the two classes.

**Key Visualizations**:

Bar plots showing the frequency of hateful vs. non-hateful comments.
A word cloud to visualize the most common words in each category (hateful and non-hateful comments).
A box plot illustrating the distribution of comment lengths across the two labels.
Findings and Insights:
Hateful comments often include specific keywords or slang, while non-hateful comments tend to be longer and more formal.
The dataset’s imbalance required further preprocessing to ensure fair model training.
Conclusion:
The EDA provided critical insights into the dataset, allowing for informed decisions in the subsequent stages of the project. By understanding the dataset’s characteristics, the preprocessing and modeling approaches were tailored to effectively address the challenges of hate speech detection.

## Primary/Machine Learning Analysis

Based on the insights from the Exploratory Data Analysis (EDA), three machine learning techniques were selected for the project: **DistilBERT**, **LSTM (Long Short-Term Memory)**, and **Naive Bayes**. These models were chosen for their effectiveness in processing and analyzing textual data, especially for Natural Language Processing (NLP) tasks such as hate speech detection.

### Why were these techniques chosen?

- **DistilBERT**: A smaller and faster version of the BERT (Bidirectional Encoder Representations from Transformers) model. It is pre-trained on large corpora and excels at understanding the context of words in sentences. DistilBERT’s transfer learning capability allows it to adapt quickly to hate speech detection with minimal computational resources.

- **LSTM**: A type of Recurrent Neural Network (RNN) effective for processing sequential data like text. It captures long-term dependencies, crucial for understanding the context of words and phrases, making it ideal for identifying subtle cues in language.

- **Naive Bayes**: A computationally efficient and straightforward model. It assumes independence between features and is useful for quick text analysis, providing a solid baseline for comparison.

### How do these techniques help answer the key questions?

- **DistilBERT**: By tokenizing and transforming text into embeddings, DistilBERT captures word relationships and context. It focuses on the critical parts of the input, ensuring accurate classification of hateful versus non-hateful comments.
  
- **LSTM**: Processes tokenized data and identifies sequential patterns, detecting relationships between words over time. This model can distinguish subtle language cues that separate hate speech from non-hateful content.
  
- **Naive Bayes**: Offers quick training and prediction times, ideal for initial exploration of text data distributions and feature importance, especially through TF-IDF.

### Implementation Details:

- **DistilBERT**: Utilized the pre-trained `distilbert-base-uncased` model. Text data was tokenized with the DistilBERT tokenizer, and data loaders were created with a batch size of 16. The AdamW optimizer was chosen for training, and the model was fine-tuned for 3 epochs.
  
- **LSTM**: The text data was tokenized and padded to ensure uniform input length. The model used ReLU and Sigmoid activation functions with the Adam optimizer. The LSTM model was trained for 5 epochs, with accuracy and loss monitored during training.
  
- **Naive Bayes**: The dataset was vectorized using the TF-IDF approach. The data was split into 80% training and 20% testing sets. The Multinomial Naive Bayes variant was used, providing an efficient but less nuanced approach compared to DistilBERT and LSTM.

### Conclusion:

- **DistilBERT** and **LSTM** were highly effective in addressing the project’s objectives. DistilBERT excelled due to its advanced contextual understanding, while LSTM provided a complementary approach by analyzing sequential relationships in text.
  
- **Naive Bayes**, while efficient and useful for initial exploration, struggled with more complex patterns in language. It was still valuable for understanding feature importance and text distributions but was outperformed by DistilBERT and LSTM in detecting nuanced language patterns.

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
