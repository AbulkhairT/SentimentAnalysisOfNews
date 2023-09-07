# Essential data processing, machine learning, and visualization libraries are imported
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
data = pd.read_csv('news_dataset.csv')

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# A two-step pipeline is set:
# TfidfVectorizer: Converts text data into numerical features using
# Term Frequency-Inverse Document Frequency.
# PassiveAggressiveClassifier: A classifier suitable for text classification
# tasks.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')), #Scale data
    ('clf', PassiveAggressiveClassifier()) # Use a classifier
])

# Define the hyperparameters and their possible values
parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'tfidf__min_df': (1, 2, 3),
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],  # unigrams or bigrams or only bigrams
    'clf__max_iter': (10, 50, 80),
    'clf__C': (0.1, 1)
}

# Search the hyperparameters
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Print the best set of hyperparameters
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Predict using the best model
y_pred = grid_search.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")

#The accuracy of the model on the test set is displayed, showing how well  the model classifies news articles.
#Visualization:
#A bar chart visualizes the model's accuracy and misclassification rates, providing a
# clear picture of the model's performance.

# Data Visualization
labels = ['Accuracy', 'Misclassification']
values = [accuracy*100, 100-accuracy*100]

# Data Visualization
# charts to show accuracy % and missclasification %
plt.figure(figsize=(7,7))
sns.barplot(x=labels, y=values, palette='husl')
plt.ylabel('% Rate')
plt.title('Model Performance Visualization')
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10)
plt.show()
