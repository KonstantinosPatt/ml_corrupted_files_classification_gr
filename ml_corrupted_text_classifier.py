import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import logging
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
# Main function
def main():

    # 1. Load the CSV dataset
    df = pd.read_excel('data/currupted_files_classification_dataset_gr.xlsx')
    logger.info(f"Loaded dataset with {len(df)} rows.")
    df["text"] = df["text"].fillna("")
    df["value"] = df["value"].fillna("")

    # 2. Split dataset into train and validation
    X = df.copy()  # All features (we'll select columns via ColumnTransformer)
    y = df['value']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    logger.info(f"Training set: {len(X_train)} rows; Validation set: {len(X_val)} rows.")

    # 3. Build a ML Pipeline with Trigrams in TF-IDF
    # Using TF-IDF for "text" (30 features) with ngram_range
    # set to (1,3) to include unigrams, bigrams, and trigrams.
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', TfidfVectorizer(max_features=30, ngram_range=(1, 3)), 'text'),
        ],
        remainder='drop'
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    # 4. Train the Model on Training Set
    model.fit(X_train, y_train)
    logger.info("Model training complete.")

    # 5. Evaluate on the Validation Set
    y_val_pred = model.predict(X_val)
    
    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_val, y_val_pred, labels=["good", "bad"])
    cr = classification_report(y_val, y_val_pred, labels=["good", "bad"])
    
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    
    logger.info("Classification Report:")
    logger.info(f"\n{cr}")

    # Save Validation Predictions to CSV
    X_val = X_val.copy()  # Create a copy so we don't modify the original
    X_val['true_label'] = y_val.values
    X_val['predicted_label'] = y_val_pred
    output_csv = "validation_predictions_corr.csv"
    X_val.to_csv(output_csv, index=False)
    logger.info(f"Validation predictions saved to {output_csv}.")

    # 6. Cross-Validation Evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    logger.info("Cross-validation accuracy scores: %s", cv_scores)
    logger.info("Mean cross-validation accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # 7.Save the model and preprocessor
    with open('corruption_classification_model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()

