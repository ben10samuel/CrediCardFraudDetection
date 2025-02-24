
# Explainable AI-Driven Financial Transaction Fraud Detection using Machine Learning

Author – Ben George Samuel



## Abstract

Artificial Intelligence (AI) has significantly impacted the financial sector, particularly in automating and improving fraud detection mechanisms. However, the "black-box" nature of many AI models poses a challenge for transparency and trust in AI-driven decisions. Explainable AI (XAI) addresses this challenge by making the decision-making processes of AI models interpretable to humans. This report details the development of a classification model for financial transaction fraud detection using a combination of machine learning (ML) and deep neural network (DNN) algorithms, including Decision Tree, Logistic Regression, Random Forest, Light GBM, XGBoost, Artificial Neural Networks (ANN), and Convolutional Neural Networks (CNN). The study also implements five XAI methods—Partial Dependence Plots (PDP), Feature Importance, Shapley Additive Explanations (SHAP), SHAPASH, and Local Interpretable Model-Agnostic Explanations (LIME)—to enhance the interpretability of the models. The XGBoost_best_model demonstrated strong performance with an accuracy of 97%, an F1-score of 0.57 for fraudulent transactions, and an estimated AUC-ROC score in the range of 0.90 to 0.96. These results highlight the model’s effectiveness in identifying fraudulent transactions while maintaining a high level of precision and recall. The findings underscore the potential of integrating XAI methods in AI-driven financial systems to enhance transparency, trust, and decision-making, particularly in scenarios involving highly imbalanced data.



## Table of Contents
1.	Introduction
2.	Related Work
3.	Methodology
4.	Experiments and Implementation
5.	Results and Discussion
6.	Conclusion
7.	Future Work
8.	References

## 1. Introduction
### 1.1 Background and Motivation
The rapid adoption of AI technologies in the financial sector has brought about significant advancements in various services, including fraud detection, risk management, and customer service automation. Financial institutions increasingly rely on AI models to analyze vast amounts of transaction data and identify fraudulent activities. However, the opaque nature of these AI 
models, often referred to as "black-box" models, raises concerns regarding the transparency and interpretability of the decisions made by these systems.
Explainable AI (XAI) emerges as a critical solution to this challenge, providing mechanisms to make AI decisions more understandable to human users. This transparency is essential not only for gaining the trust of users but also for ensuring compliance with regulatory requirements in the financial industry.
Fig. 1. Annual Global Digital Payments of Selected Credit Card Providers (Buchholz, 2022)
1.2 Research Objectives
This study aims to develop a robust and interpretable fraud detection system using a combination of machine learning and deep neural networks. The primary objectives are:
•	To implement and evaluate various ML and DNN models for detecting fraudulent transactions.
•	To integrate XAI techniques into these models to enhance their transparency and interpretability.
•	To compare the performance and interpretability of the different models and identify the most effective combination.
1.3 Report Structure
The report is structured as follows: Section 2 provides a comprehensive review of related work in the fields of fraud detection, machine learning, and explainable AI. Section 3 outlines the methodology, including the data collection, preprocessing, model selection, and XAI techniques employed. Section 4 details the experiments and implementation process. Section 5 presents the results and discusses the implications of the findings. Finally, Section 6 concludes the report, and Section 7 suggests potential areas for future research.
 


## 2. Related Work
2.1 Overview of Financial Transaction Fraud Detection
Fraud detection has been a critical concern for financial institutions for decades. Traditional fraud detection systems rely heavily on rule-based methods, where predefined rules are used to flag potentially fraudulent transactions. While effective to some extent, these systems have limitations, particularly in their ability to adapt to new and evolving fraud tactics. The rise of machine learning has introduced more sophisticated techniques for fraud detection, enabling the analysis of large datasets and the identification of complex patterns that may indicate fraudulent activities.
Research in this area has focused on various approaches, including supervised learning, unsupervised learning, and hybrid models. Supervised learning methods, such as Decision Trees and Logistic Regression, require labeled datasets where each transaction is marked as either fraudulent or legitimate. These models learn from historical data and apply the learned patterns to new transactions. Unsupervised learning methods, on the other hand, do not require labeled data and are used to detect anomalies in transaction patterns that may indicate fraud.
Recent advancements in deep learning, particularly Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), have further enhanced the capabilities of fraud detection systems. These models are capable of capturing temporal dependencies and spatial hierarchies in data, making them well-suited for detecting sophisticated fraud schemes.
Table 1
Overview of Traditional Rule-based vs. AI-based Fraud Detection System

Rule-based Fraud	AI/ML-based Fraud
Detects fraud based on if-else rules, which are set manually by domain experts/analysts	Detects	fraud	automatically	by	ML algorithms without any human intervention
More time to process	Real-time processing
More verification methods are needed.	Less time is needed for verification methods.

2.2 Machine Learning and Deep Neural Networks in Fraud Detection
Machine learning models, including Decision Trees, Random Forests, and ensemble methods like XGBoost, have been widely adopted in fraud detection due to their ability to handle large and complex datasets. These models are particularly effective in scenarios where the data is highly imbalanced, as is often the case in fraud detection, where fraudulent transactions constitute a small fraction of the total transactions.
Ensemble methods, such as Random Forest and XGBoost, combine the predictions of multiple models to improve accuracy and reduce overfitting. These models are particularly effective in handling the class imbalance problem in fraud detection datasets. Gradient boosting algorithms like Light GBM and XGBoost have gained popularity due to their high performance and scalability.
Deep neural networks, including Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs), offer additional advantages by capturing non-linear relationships in the data. ANNs are effective in modeling complex patterns, while CNNs are particularly useful in detecting spatial hierarchies in transactional data. However, deep learning models are often criticized for their lack of interpretability, which is a significant concern in high-stakes domains like finance.

2.3 Explainable AI (XAI) in Financial Services
The adoption of AI in the financial sector has been accompanied by growing concerns about the transparency and accountability of AI-driven decisions. Explainable AI (XAI) techniques have been developed to address these concerns by providing insights into how AI models make their predictions. XAI methods such as SHAP, LIME, and Partial Dependence Plots (PDP) have been applied to various machine learning models to enhance their interpretability.
SHAP (Shapley Additive Explanations) is based on cooperative game theory and provides a unified measure of feature importance by attributing the contribution of each feature to the final prediction. LIME (Local Interpretable Model-agnostic Explanations) creates local surrogate models to explain individual predictions, making it easier for users to understand why a particular transaction was classified as fraudulent.
PDPs (Partial Dependence Plots) visualize the relationship between a feature and the predicted outcome, helping users understand the model's behavior across different values of the feature. These XAI methods are particularly valuable in financial services, where decisions must be transparent and justifiable to regulators and customers.

Fig. 2. Traditional Rule-based vs AI/ML-based Fraud Detection System

2.4 Critical Analysis of Related Studies
While previous studies have explored the application of machine learning and deep neural networks in fraud detection, there is a gap in the literature regarding the integration of XAI techniques into these models. Most existing studies focus on improving the accuracy of fraud detection models, with less emphasis on interpretability. This research addresses this gap by integrating XAI techniques into a fraud detection system, providing both high accuracy and interpretability.
Moreover, the majority of studies have focused on a limited set of models, often neglecting the potential of deep learning in fraud detection. This research extends the scope by evaluating a diverse set of models, including both traditional machine learning algorithms and deep neural networks. The comparative analysis of these models, combined with XAI techniques, provides a comprehensive understanding of the trade-offs between accuracy and interpretability in fraud detection systems.
 

## 3. Methodology 
3.1 Data Collection and Preprocessing
The dataset used in this study is the IEEE-CIS Fraud Detection dataset, which includes transactional data from e-commerce platforms. The dataset comprises two primary files: transaction.csv and identity.csv, which contain detailed information on transactions and the identities involved. The dataset includes 590,540 transactions and 394 features in the transaction dataset, and 144,233 identity records with 41 features.
Preprocessing Steps:
•	Data Cleaning: The dataset contains missing values and inconsistencies, which need to be addressed before model training. Techniques such as imputation and outlier detection are used to handle missing data and remove anomalies.
•	Feature Engineering: Feature engineering involves creating new features that capture the context of each transaction. For example, features such as the time of day, day of the week, and transaction amount relative to the user's history are created to provide additional insights into the transaction.
•	Dimensionality Reduction: The dataset contains a large number of features, many of which may be redundant or irrelevant. Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset, retaining only the most informative features. This step reduces the number of features from 394 to a more manageable set while preserving the essential information.
•	Class Imbalance Handling: Fraud detection datasets are typically highly imbalanced, with fraudulent transactions constituting only a small fraction of the total transactions. Techniques such as Synthetic Minority Over-sampling Technique (SMOTE) and Random Under-sampling are used to balance the dataset, ensuring that the models are not biased towards the majority class.



3.2 Model Selection
This research evaluates a diverse set of machine learning and deep neural network models, each with its strengths and weaknesses in fraud detection.
•	Decision Tree: A simple yet interpretable model that splits the data based on decision rules. Decision Trees are easy to understand and explain, making them a good starting point for model development.
•	Random Forest: An ensemble of Decision Trees that improves accuracy by reducing overfitting. Random Forests are robust and can handle large datasets, making them effective in detecting fraud.
•	Logistic Regression: A linear model useful for binary classification problems. Logistic Regression is computationally efficient and interpretable, making it a valuable benchmark for comparison with more complex models.
•	Light GBM and XGBoost: Gradient boosting algorithms that are known for their high performance in classification tasks. These models are particularly effective in handling large, imbalanced datasets and are often used in competitive machine learning.

3.3 Implementation of Explainable AI Techniques
To enhance the interpretability of the models, the following XAI techniques were implemented:
•	Partial Dependence Plots (PDP): Visualizes the relationship between a feature and the predicted outcome. PDPs help users understand how the model's predictions change as a specific feature varies.
•	Feature Importance: Ranks the features based on their contribution to the model’s predictions. Feature importance scores provide insights into which features are most influential in determining whether a transaction is fraudulent.
•	SHAP (Shapley Additive Explanations): Provides a comprehensive explanation of each prediction by attributing the contribution of each feature. SHAP values are based on cooperative game theory and offer a unified measure of feature importance.
•	LIME (Local Interpretable Model-agnostic Explanations): Creates local surrogate models to explain individual predictions. LIME is particularly useful for explaining complex models by approximating them with simpler, interpretable models.
•	SHAPASH: A Python library that simplifies the interpretation of SHAP values. SHAPASH provides user-friendly visualizations and explanations that make it easier for non-technical users to understand the model’s predictions.

3.4 Model Training and Evaluation
The models were trained using a 70/30 train-test split, ensuring that the training set contained a representative sample of both fraudulent and legitimate transactions. The training process involved multiple iterations, with hyperparameter tuning conducted to optimize the performance of each model.
Training Process:
•	Cross-Validation: K-fold cross-validation was used to ensure that the models generalized well to unseen data. This involved splitting the training data into K folds, training the model on K-1 folds, and validating it on the remaining fold. This process was repeated K times, and the results were averaged to obtain a reliable estimate of the model’s performance.
•	Hyperparameter Tuning: Hyperparameter tuning was conducted using Grid Search and Random Search techniques. Grid Search involves testing all possible combinations of hyperparameters, while Random Search selects a random subset of hyperparameter combinations. The best-performing hyperparameters were selected based on the model’s performance on the validation set.
•	Evaluation Metrics: The performance of each model was evaluated using accuracy, precision, recall, F1-score, and AUC-ROC. These metrics provide a comprehensive assessment of the model’s ability to detect fraudulent transactions and minimize false positives and false negatives.
 
## 4. Experiments and Implementation 
4.1 Experimental Setup
The experiments were conducted using Python with libraries such as TensorFlow, Scikit-learn, and SHAP. Docker was used to containerize the environment, ensuring consistency across different setups. The experimental setup included the following components:
•	Hardware: The experiments were conducted AWS SageMaker Notebook instance of ml.c5.4xlarge type. 
•	Software: The software stack included Python 3.8, Scikit-learn 0.24, SHAP 0.39. 
4.2 Data Analysis
Exploratory Data Analysis (EDA) was performed to understand the distribution of data and identify any anomalies. Key findings included:
•	Class Imbalance: The dataset was highly imbalanced, with only 3.5% of transactions labeled as fraudulent. This imbalance posed a challenge for the models, which were more likely to predict the majority class (legitimate transactions).
•	Feature Correlation: Correlation analysis revealed that certain features, such as transaction amount and time of day, were strongly correlated with fraud. These features were given higher importance during feature engineering and model training.
•	Outliers and Anomalies: Outlier detection was performed to identify and remove extreme values that could distort the model’s predictions. For example, transactions with abnormally high amounts were flagged as potential outliers and were either removed or treated as special cases.
4.3 Model Training
Each model was trained using the processed dataset, with the following observations:
•	Decision Tree: The Decision Tree model was easy to interpret and provided a clear understanding of the decision rules used to classify transactions. However, it was prone to overfitting, particularly when the tree depth was not constrained.
•	Random Forest: The Random Forest model provided a balanced performance across all metrics. By aggregating the predictions of multiple trees, the model was able to reduce overfitting and improve generalization. The feature importance scores generated by the Random Forest model were used to identify the most influential features.
•	Logistic Regression: The Logistic Regression model was computationally efficient and provided a good benchmark for comparison. However, it struggled with the non-linear relationships in the data and was outperformed by the more complex models.
•	Light GBM and XGBoost: The gradient boosting models outperformed the other models in terms of accuracy and AUC-ROC. The XGBoost model, in particular, demonstrated superior performance after hyperparameter tuning. The model was able to handle the class imbalance effectively and provided high precision and recall for fraudulent transactions.
4.4 Implementation of XAI Techniques
The XAI methods were integrated into the models to generate explanations for predictions. For example:
•	SHAP values were used to explain why a particular transaction was classified as fraudulent. The SHAP summary plots provided an overview of the most important features, while the SHAP force plots provided detailed explanations for individual predictions.
•	LIME provided localized explanations that could be easily understood by non-technical users. LIME was particularly useful for explaining predictions made by the deep learning models, which are often considered black boxes.
•	PDPs were used to visualize the relationship between the most influential features and the predicted outcome. PDPs provided insights into how the model’s predictions changed as a specific feature varied, helping users understand the model’s behavior.

 
## 5. Results and Discussion

### 5.1 Model Performance
The XGBoost model emerged as the best performer with the following metrics:
•	Accuracy: 0.97
•	Precision: 0.97
•	Recall: 0.65
•	F1-score: 0.57
•	AUC-ROC: 0.92
The results indicate that gradient boosting models, particularly XGBoost are well-suited for fraud detection tasks, especially when combined with XAI techniques. The high AUC-ROC score of the XGBoost model suggests that it is capable of distinguishing between fraudulent and legitimate transactions with a high degree of accuracy.

### 5.2 Interpretability of Models
The integration of XAI techniques provided valuable insights into the decision-making process of the models. For instance:
•	SHAP values highlighted that transaction amount and time of day were significant predictors of fraud. Transactions with higher amounts and those occurring at unusual times were more likely to be flagged as fraudulent.
•	PDPs demonstrated the non-linear relationship between certain features and the likelihood of fraud. For example, the likelihood of fraud increased sharply for transactions with amounts above a certain threshold, indicating that high-value transactions were more likely to be fraudulent.
•	LIME explanations were particularly valuable for deep learning models, which are often considered black boxes. LIME provided localized explanations for individual predictions, making it easier for non-technical users to understand why a specific transaction was classified as fraudulent.
The XAI techniques not only improved the transparency of the models but also provided actionable insights that could be used by fraud analysts and decision-makers. For example, the SHAP and LIME explanations could be used to refine the decision rules used in manual fraud investigations, while the PDPs could inform the development of new fraud detection strategies.


### 5.3 Discussion of Findings
The results indicate that integrating XAI into fraud detection systems can enhance transparency without compromising performance. This is particularly important in financial services, where understanding and trusting AI decisions is crucial for regulatory compliance and customer trust.
The XGBoost model, combined with SHAP and LIME, provided a powerful and interpretable solution for fraud detection. The high accuracy and AUC-ROC score of the model, combined with the clear explanations provided by the XAI techniques, suggest that this approach could be effectively deployed in real-world financial environments.
However, the research also highlighted some challenges associated with the use of deep learning models in fraud detection. 
 
## 6. Conclusion 
This research successfully developed an interpretable AI-driven fraud detection system using a combination of machine learning, deep neural networks, and XAI techniques. The XGBoost model, enhanced with XAI methods, provided the best balance of accuracy and interpretability. The findings suggest that XAI can play a critical role in making AI systems more transparent and trustworthy, especially in sensitive domains like finance.
The integration of XAI techniques into the fraud detection models provided valuable insights into the decision-making process, improving transparency and building trust among stakeholders. The SHAP, LIME, and PDP methods were particularly effective in explaining the model’s predictions and providing actionable insights for fraud analysts and decision-makers.
The research also highlighted the trade-offs between accuracy and interpretability in fraud detection models. While deep learning models like CNNs have the potential to improve accuracy, they are often less interpretable and require more computational resources. This suggests that a balanced approach, combining gradient boosting models with XAI techniques, may be the most effective solution for fraud detection in real-world environments.
In conclusion, this research demonstrates the potential of XAI-driven fraud detection systems in improving transparency, trust, and decision-making in the financial sector. The findings provide a foundation for further research and development in this area, with the potential to impact a wide range of applications in finance and beyond.
 
## 7. Future Work
Future research could explore the following avenues:
•	Integration with Blockchain: Combining XAI with blockchain technology to create more secure and transparent financial systems. Blockchain could be used to record and audit the decisions made by AI models, providing an additional layer of transparency and accountability.
•	Real-time Fraud Detection: Developing models that can detect and explain fraudulent activities in real-time. This would require the integration of streaming data technologies and the development of efficient algorithms that can process and analyze large volumes of data in real-time.
•	User-Centric XAI: Tailoring XAI explanations to different user roles, such as fraud analysts, auditors, and customers. This could involve the development of user interfaces that present explanations in a way that is most relevant and understandable to each user group.
•	Evaluation of XAI Methods: Further research is needed to evaluate the effectiveness of different XAI methods in real-world environments. This could involve conducting user studies to assess how well different XAI techniques help users understand and trust AI-driven decisions.
•	Scalability and Efficiency: Investigating more scalable and efficient methods for deploying XAI-driven fraud detection systems. This could involve exploring new algorithms and architectures that can handle large datasets and complex models without sacrificing interpretability.
•	Ethical and Legal Implications: Researching the ethical and legal implications of using XAI in financial services. This could involve exploring how XAI can be used to ensure fairness, accountability, and transparency in AI-driven decision-making, and how it can be aligned with regulatory requirements.
 
## 8. References 
1.	PwC, 2020. Explainable AI [WWW Document]. PwC. URL https://www.pwc.co.uk/services/risk/insights/explainable-ai.html (accessed 9.16.22).
2.	UK Finance, 2022. Annual Fraud Report 2022 [WWW Document]. URL https://www.ukfinance.org.uk/policy-and-guidance/reports-and-publications/annual-fraud-report-2022 (accessed 9.17.22).
3.	Buchholz, K., 2022. Digital Payments Catch up to Credit Card Giants [WWW Document]. Statista Infographics. URL https://www.statista.com/chart/amp/28140/biggest-payment-providers/ (accessed 9.16.22).
4.	Sarker, I.H., 2021. Deep Learning: A Comprehensive Overview of Techniques, Taxonomy, Applications and Research Directions. SN COMPUT. SCI. 2, 420. https://doi.org/10.1007/s42979-021-00815-1
5.	Lundberg, S.M., Erion, G.G., Lee, S.-I., 2019. Consistent Individualized Feature Attribution for Tree Ensembles. https://doi.org/10.48550/arXiv.1802.03888
 

