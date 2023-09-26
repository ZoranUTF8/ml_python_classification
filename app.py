import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Specify the file path relative to your script
file_path = 'data/music.csv'

# Read the dataset into a DataFrame
df = pd.read_csv(file_path)

# Split data into input (features) and output (target) sets
# Input set (X): Contains all columns except "genre"
X = df.drop(columns=["genre"])

# Output set (Y): Contains only the "genre" column
Y = df["genre"]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Create a DecisionTreeClassifier model
model = DecisionTreeClassifier()

# Train the model on the training data to learn patterns
model.fit(X_train, Y_train)

# Make predictions on the testing data
predictions = model.predict(X_test)

# Evaluate the model's accuracy by comparing predicted labels to actual labels
ac = accuracy_score(Y_test, predictions)

# Print the accuracy score
print(ac)
