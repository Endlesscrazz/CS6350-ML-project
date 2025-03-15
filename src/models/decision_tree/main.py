import argparse
from data_loader import load_data
from decision_tree import build_tree, predict
from evaluation import compute_metrics
from cross_validation import cross_validations, grid_search_cv

def tune_hyperparameters(X_train, y_train):

    max_depth_values = [5,10,15]
    min_samples_split_values = [2,5,7]

    best_params, best_f1 = grid_search_cv(X_train, y_train, max_depth_values, min_samples_split_values, k=5)
    print(f"Best hyperparameters found: {best_params} with F1 score: {best_f1:.3f}")
    return best_params

def main():

    #Command line argument parser
    parser = argparse.ArgumentParser(description="Train Decision Tree and perform cross-validation with hyperparameter tuning")
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning.')
    parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth of the decision tree.')
    parser.add_argument('--min_samples_split', type=int, default=5, help='Minimum samples required to split a node.')
    parser.add_argument('--k', type=int, default=5, help='Number of folds for cross-validation.')
    args = parser.parse_args()
    
    #Load training data
    X_train, y_train = load_data('data/train.csv', label_column='label')

    if args.tune:
        best_params = tune_hyperparameters(X_train, y_train)
        max_depth = best_params["max_depth"]
        min_samples_split = best_params["min_samples_split"]

    avg_precision, avg_recall, avg_f1_score = cross_validations(X_train, y_train, k=args.k, max_depth=max_depth, min_samples_split=min_samples_split)
    print(f"Cross-Validation Metrics -> Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1-score: {avg_f1_score:.3f}")

    tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
    print("Decision Tree built")

    train_preds = predict(tree, X_train)
    precision, recall, f1 = compute_metrics(y_train, train_preds)
    print(f"Training Metrics -> Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f} ")

    #Load test data
    X_test, y_test = load_data('data/test.csv', label_column='label')

    test_preds = predict(tree, X_test)

    if y_test is not None:
        precision, recall, f1 = compute_metrics(y_test, test_preds)
        print(f"Test Metrics -> Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f} ")

    # output_df = pd.DataFrame({'prediction': test_preds})
    # output_df.to_csv('/output/predictions.csv')
    # print("Test predcitions saved succesfully")

if __name__ == '__main__':
    main()
