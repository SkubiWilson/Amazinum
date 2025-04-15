
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class titanic:
    def __init__(self):
        self.train_dt = pd.read_csv('data/train.csv')
        self.test_dt = pd.read_csv('data/test.csv')

    def data_pred(self):
        self.train_dt['Age'].fillna(self.train_dt['Age'].median(), inplace=True)
        self.train_dt['Embarked'].fillna(self.train_dt['Embarked'].mode()[0], inplace=True)
        self.train_dt['Sex'] = self.train_dt['Sex'].map({'male': 0, 'female': 1})
        self.train_dt['Embarked'] = self.train_dt['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        self.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.X = self.train_dt[self.features]
        self.y = self.train_dt['Survived']

    def model_df(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=44)
        self.model = RandomForestClassifier(n_estimators=100, random_state=44)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_val)
        print("Accuracy:", accuracy_score(self.y_val, y_pred))

    def survivor(self):
        self.test_dt['Age'].fillna(self.train_dt['Age'].median(), inplace=True)
        self.test_dt['Fare'].fillna(self.train_dt['Fare'].median(), inplace=True)
        self.test_dt['Sex'] = self.test_dt['Sex'].map({'male': 0, 'female': 1})  # <-- виправлення
        self.test_dt['Embarked'] = self.test_dt['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        X_test = self.test_dt[self.features]
        predictions = self.model.predict(X_test)
        submission = pd.DataFrame({
            'PassengerId': self.test_dt['PassengerId'],
            'Survived': predictions
        })
        submission.to_csv('submss.csv', index=False)

        print(submission.head(10))





launch_ship = titanic()
launch_ship.data_pred()
launch_ship.model_df()
launch_ship.survivor()