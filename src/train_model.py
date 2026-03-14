from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        print(name, "Accuracy:", score)

        if score > best_score:
            best_score = score
            best_model = model

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.clf()

    probs = best_model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.title("ROC Curve (AUC=%0.2f)" % roc_auc)
    plt.savefig("outputs/roc_curve.png")
    plt.clf()

    joblib.dump(best_model, "models/churn_model.pkl")