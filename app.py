from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("C:\\Users\\manas\\OneDrive\\Documents\\AI_Final_project\\seattle-weather.csv")

# Ensure there are no null values in our data
data.info()

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['weather', 'date'])  # Features
y = data['weather']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# hypertuning for improving accuracy
# Define the parameters grid for grid search
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'criterion': ['gini', 'entropy']
}

# Initialize the decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters found by grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the decision tree model with the best parameters
best_decision_tree = DecisionTreeClassifier(**best_params)
best_decision_tree.fit(X_train, y_train)

# Evaluate the model
y_pred_train = best_decision_tree.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", train_accuracy)

y_pred_test = best_decision_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Testing Accuracy:", test_accuracy)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the request
    precipitation = float(request.form['precipitation'])
    temp_max = float(request.form['temp_max'])
    temp_min = float(request.form['temp_min'])
    wind = float(request.form['wind'])

    # Make predictions
    new_data = pd.DataFrame({
        'precipitation': [precipitation],
        'temp_max': [temp_max],
        'temp_min': [temp_min],
        'wind': [wind]
    })
    predicted_weather = best_decision_tree.predict(new_data)
    return render_template('result.html', predicted_weather=predicted_weather[0])

if __name__ == '__main__':
    app.run(debug=True)
