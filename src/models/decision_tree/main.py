import argparse
from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.cross_validation import cross_validations_generic, grid_search_cv_generic

from models.decision_tree.builders import build_model_decision_tree
# def tune_hyperparameters(X_train, y_train):

#     max_depth_values = [5,10,15]
#     min_samples_split_values = [2,5,7]

#     best_params, best_f1 = grid_search_cv_generic(X_train, y_train, max_depth_values, min_samples_split_values, k=5)
#     print(f"Best hyperparameters found: {best_params} with F1 score: {best_f1:.3f}")
#     return best_params

def main():

    #Command line argument parser
    parser = argparse.ArgumentParser(description="Train Decision Tree and perform cross-validation with hyperparameter tuning")
    parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth of the decision tree.')
    parser.add_argument('--min_samples_split', type=int, default=5, help='Minimum samples required to split a node.')
    args = parser.parse_args()
    
    #Load training data
    X_train, y_train = load_data('data/train.csv', label_column='label')

    #Load test data
    X_test, y_test = load_data('data/test.csv', label_column='label')

    hyperparams = {"max_depth": args.max_depth, "min_samples_split": args.min_samples_split }

    model = build_model_decision_tree(X_train, y_train, hyperparams)

    #Evaluation on training data
    train_preds = model.predict(X_train)
    train_prec, train_rec, train_f1 = compute_metrics(y_train, train_preds)
    print(f"Training Metrics -> Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    if y_test is not None:
        test_preds = model.predict(X_test)
        precision, recall, f1 = compute_metrics(y_test, test_preds)
        print(f"Test Metrics -> Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f} ")

if __name__ == '__main__':
    main()
