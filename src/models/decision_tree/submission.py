import pandas as pd
import argparse
from data_loader import load_data
from decision_tree import build_tree, predict

def create_submission(max_depth, min_samples_split):
    
    X_eval, _ = load_data('data/eval.anon.csv', label_column='label')  
    #print(X_eval.shape)

    X_train, y_train = load_data('data/train.csv', label_column='label')
    tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
    
    eval_preds = predict(tree, X_eval)
    
    eval_ids = pd.read_csv('data/eval.id', header=None, names=["example_id"])
    
    submission = pd.DataFrame({
        "example_id": eval_ids["example_id"],
        "label": eval_preds
    })
    
    # Saving the submission file with required header
    submission.to_csv('output/submission.csv', index=False)
    print("Submission file saved to output/submission.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate submission file with tuned decision tree.")
    parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth of the decision tree.')
    parser.add_argument('--min_samples_split', type=int, default=5, help='Minimum samples required to split a node.')
    
    args = parser.parse_args()
    create_submission(args.max_depth, args.min_samples_split)
