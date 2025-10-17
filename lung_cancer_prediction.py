import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Loading the survey lung cancer.csv
csvdata = pd.read_csv('\survey lung cancer.csv')

# csvdata.info() "to check null values"

# Encoding the features
label_encoder = LabelEncoder()
for column in csvdata.columns:
    csvdata[column] = label_encoder.fit_transform(csvdata[column])

# csvdata.head() "to check the convertions 0/1"
  
# Separating features (X) and target (y)
# The last column is LUNG_CANCER(our target)
X = csvdata.drop('LUNG_CANCER', axis=1)
y = csvdata['LUNG_CANCER']  

#Split data into 80% Training and 20% Testing Sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
