# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.tree import DecisionTreeClassifier  # For decision tree classifier
import joblib  # For model and data storage
from sklearn import tree  # For tree visualization

# Specify the file path relative to your script
file_path = 'data/music.csv'

# Read the dataset into a DataFrame using pandas
df = pd.read_csv(file_path)

# Split the data into input (features) and output (target) sets
# Input set (X): Contains all columns except "genre"
X = df.drop(columns=["genre"])

# Output set (Y): Contains only the "genre" column
Y = df["genre"]

# Create a DecisionTreeClassifier model
model = DecisionTreeClassifier()

# Train the model on the training data to learn patterns
model.fit(X, Y)

# Create a visualization of the decision tree model and save it as a .dot file
# This visualization can be converted to an image using Graphviz or other tools
tree.export_graphviz(model, out_file="music_recomender.dot", feature_names=["age", "gender"],
                     class_names=sorted(Y.unique()), label="all", rounded=True, filled=True)

# Save the trained model to a file (not currently in use)
# joblib.dump(model, 'music_recomender.joblib')

# Load a previously trained model from a file (not currently in use)
# model = joblib.load('music_recomender.joblib')

# Make predictions using the trained model on a sample input
predictions = model.predict([[31, 1]])

# Print the predictions
print(predictions)
