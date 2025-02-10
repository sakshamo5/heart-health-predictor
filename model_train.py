import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
mail_data = pd.read_csv("mail_data.csv")
mail_data.dropna(inplace=True)

# Ensure correct Category encoding
mail_data["Category"] = mail_data["Category"].map({"spam": 1, "ham": 0})

# Split into features and Categorys
X = mail_data["Message"]  # Change column name if needed
y = mail_data["Category"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\n✅ Model and vectorizer saved successfully!")
