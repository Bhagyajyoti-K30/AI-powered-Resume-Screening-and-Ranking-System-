# AI-powered-Resume-Screening-and-Ranking-System-

#Import necessary Python libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample data
data = {
    'resume': [
        "Experienced software engineer with skills in Python, Java, and machine learning.",
        "Project manager with a background in software development and team leadership.",
        "Data analyst proficient in SQL, Python, and data visualization."
    ],
    'label': [1, 0, 1]  # 1 = relevant, 0 = not relevant
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocessing and feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['resume'])
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))


An AI-powered resume screening and ranking system can help automate the process of evaluating job applications. This system typically uses natural language processing (NLP) and machine learning techniques to analyze resumes and rank them based on their relevance to a job description.
