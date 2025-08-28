import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

def main():
    df = pd.read_csv("/home/parijat/machine_learning/Data/features_30_sec.csv")
    y = df['label']
    X = df.drop(columns=['label', 'filename'])
    X = X.select_dtypes(include=[np.number])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.1, stratify=y_encoded, random_state=42
    )

    classes = np.unique(y_train_full)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_full)
    scale_pos_weight_array = {i: w for i, w in zip(classes, class_weights)}

    rock_idx = np.where(label_encoder.classes_ == 'rock')[0][0]
    reggae_idx = np.where(label_encoder.classes_ == 'reggae')[0][0]

    scale_pos_weight_array[rock_idx] *= 1.3
    scale_pos_weight_array[reggae_idx] *= 1.2

    sample_weights = np.array([scale_pos_weight_array[label] for label in y_train_full])

    X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
        X_train_full, y_train_full, sample_weights, test_size=0.1, stratify=y_train_full, random_state=42
    )

    train_pool = Pool(X_train, y_train, weight=sw_train)
    val_pool = Pool(X_val, y_val, weight=sw_val)

    param_grid = {
        'depth': [4, 5],
        'learning_rate': [0.03, 0.05],
        'iterations': [1000],
        'l2_leaf_reg': [7, 10],
        'bagging_temperature': [0.5, 1.0],
        'random_strength': [1, 2]
    }

    best_acc = 0
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        model = CatBoostClassifier(
            **params,
            boosting_type='Ordered',
            eval_metric='Accuracy',
            early_stopping_rounds=100,
            task_type='CPU',
            devices='0',
            random_seed=42,
            verbose=False
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_model = model

    print(f"Best Parameters: {best_params}")
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    with open("catboost_genre_classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    evals_result = best_model.get_evals_result()
    train_acc_line = evals_result['learn']['Accuracy']
    val_acc_line = evals_result['validation']['Accuracy']

    plt.figure(figsize=(10,6))
    plt.plot(train_acc_line, label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(val_acc_line, label='Validation Accuracy', color='green', linewidth=2)
    plt.axhline(y=test_acc, color='red', linestyle='--', label='Test Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation vs Test Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

