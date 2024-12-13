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

## Visualization

In this section, we present key visualizations that provide insights into the analysis, preprocessing, and performance of the hate speech detection models.

1. **Text Length Distribution by Label**  
   This plot shows the distribution of the number of words in texts labeled as **Hate Speech** and **Non-Hate Speech**, indicating that **Non-Hate Speech** tends to have longer text lengths.  
   **Figure 1**: Text Length Distribution by Label

2. **Cumulative Distribution of Text Lengths by Label**  
   This plot illustrates the cumulative percentage of text lengths, showing that shorter text lengths dominate in both categories.  
   **Figure 2**: Cumulative Distribution of Text Lengths by Label

3. **Mean Sentiment Score by Speech Type**  
   This plot compares the **mean sentiment scores** for **Hate Speech** and **Non-Hate Speech**, with error bars indicating statistical significance.  
   **Figure 3**: Mean Sentiment Score by Speech Type with Error Bars

4. **Contingency Table: Label vs Sentiment Category**  
   This heatmap displays the relationship between text labels and sentiment categories, highlighting significant differences across categories.  
   **Figure 4**: Contingency Table: Label vs Sentiment Category

5. **Distribution of Sentiment Scores**  
   This histogram shows the **distribution of sentiment scores** for both **Hate Speech** and **Non-Hate Speech**, emphasizing patterns in sentiment polarity.  
   **Figure 5**: Distribution of Sentiment Scores: Hate Speech vs Non-Hate Speech

6. **Model Performance Comparison**  
   This plot compares the performance metrics (Accuracy, Precision, Recall, F1-Score) of **Naive Bayes**, **LSTM**, and **DistilBERT**, with **DistilBERT** emerging as the best model.  
   **Figure 6**: Model Performance Comparison



## Insights and Conclusions

This project focused on **hate speech detection** using advanced **Natural Language Processing (NLP)** techniques, specifically **DistilBERT**, **LSTM**, and **Naive Bayes** models. By combining modern machine learning methodologies with robust data preprocessing and analysis, the project successfully addressed the challenges posed by real-world social media text. These efforts demonstrate how automation can play a vital role in moderating online content and promoting safer digital spaces.

### For an Uninformed Reader:
This project provides a clear understanding of the problem of hate speech detection and the steps involved in solving it. An uninformed reader gains insights into:
- The significance of identifying and addressing hate speech to combat online abuse and foster inclusive communication.
- How datasets are prepared for analysis, including data cleaning, filtering, and preprocessing to ensure quality inputs for machine learning models.
- The importance of choosing effective machine learning models, such as **DistilBERT**, **LSTM**, and **Naive Bayes**, to handle the complexities of textual data.

Through detailed explanations and visualizations, the report equips an uninformed reader with foundational knowledge about hate speech detection using machine learning.

### For a Reader Familiar with the Topic:
A knowledgeable reader gains valuable insights from this project, particularly in comparing the effectiveness of different NLP models. Key takeaways include:
- The comparative analysis of **DistilBERT**, **LSTM**, and **Naive Bayes** models, highlighting their respective strengths in handling text classification tasks.
- Strategies for addressing challenges such as data imbalance, ensuring fair and unbiased model training.
- The detailed evaluation of models using performance metrics like **accuracy**, **loss trends**, **confusion matrices**, and **AUROC curves**, which provide a nuanced understanding of their capabilities.

These insights offer advanced readers a practical perspective on implementing and evaluating machine learning models for hate speech detection.

### Key Conclusions:
- **DistilBERT** proved to be the most effective model, demonstrating superior contextual understanding and accuracy compared to **LSTM** and **Naive Bayes**.
- The incorporation of advanced preprocessing techniques, such as **language filtering** and **sequence padding**, further enhanced model performance.
- This project emphasizes that combining state-of-the-art NLP models with rigorous preprocessing is essential for tackling hate speech detection challenges.

## Data Science Ethics

The ethical considerations in this project focused on ensuring fairness, transparency, and the mitigation of biases in hate speech detection, given its sensitive nature.

### Potential Biases in Data Collection:
The dataset, sourced from open-access platforms, consisted of social media text labeled as hateful or non-hateful. Concerns included potential biases in the labeling process due to subjective human judgments, as well as societal biases in the data, such as overrepresentation of certain demographics or linguistic styles.

### Mitigation Strategies for Bias:
To address these concerns:
- The dataset was reviewed for label balance between hateful (1) and non-hateful (0) comments.
- Preprocessing steps, including duplicate removal and language filtering using the `langdetect` library, were applied to eliminate irrelevant data.
- The diversity of text inputs, including slang and emoticons, was maintained to ensure the model's generalization capability.

### Fairness in Model Development:
Machine learning models (DistilBERT, LSTM, and Naive Bayes) were chosen for their ability to handle diverse text data and imbalanced datasets. Model performance was monitored through accuracy, loss, and confusion matrices to ensure unbiased predictions and avoid overfitting or underfitting.

### Transparency in Analysis:
All preprocessing steps, model parameters, and evaluation criteria were fully documented in the report and GitHub repository. Visualizations like word clouds and AUROC curves were included to provide interpretable results, ensuring the methodology is reproducible and transparent.

### Final Thoughts:
This project bridges theory and practice in **hate speech detection**, offering a machine learning framework that addresses social media challenges while fostering safer, more inclusive digital spaces.

---

## License
[Include license information.]
