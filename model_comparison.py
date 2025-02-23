import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
from itertools import cycle

def optimize_hyperparameters(X_train, y_train):
    # Define parameter grids for each model
    param_grids = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.1, 0.5, 1.0],
                'algorithm': ['SAMME']
            }
        }
    }

    best_models = {}
    
    for name, config in param_grids.items():
        print(f"\nOptimizing {name}...")
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_models

def plot_detailed_metrics(models, X_test, y_test, save_prefix='detailed_metrics'):
    # Get feature names from X_test (assuming it's a DataFrame)
    feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
    
    # Set style for better visualization
    plt.style.use('default')
    
    # 1. ROC Curves with enhanced styling
    plt.figure(figsize=(12, 8), facecolor='white')
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', 
                color=color, lw=2.5, alpha=0.8)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{save_prefix}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Precision-Recall Curves with enhanced styling
    plt.figure(figsize=(12, 8), facecolor='white')
    
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
        avg_precision = np.mean(precision)
        plt.plot(recall, precision, label=f'{name} (Avg Precision = {avg_precision:.3f})',
                color=color, lw=2.5, alpha=0.8)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, pad=20)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.savefig(f'{save_prefix}_precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Performance Metrics Heatmap with enhanced styling
    metrics_data = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        metrics_data[name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Accuracy': accuracy
        }
    
    metrics_df = pd.DataFrame(metrics_data).T
    
    plt.figure(figsize=(12, 8), facecolor='white')
    sns.heatmap(metrics_df, annot=True, cmap='RdYlGn', fmt='.3f',
                center=0.7, vmin=0.6, vmax=1.0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Performance Metrics Comparison', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Feature Importance Bar Plot (for tree-based models)
    tree_based_models = {name: model for name, model in models.items() 
                        if hasattr(model, 'feature_importances_')}
    
    if tree_based_models:
        plt.figure(figsize=(12, 6), facecolor='white')
        bar_width = 0.8 / len(tree_based_models)
        
        for idx, (name, model) in enumerate(tree_based_models.items()):
            importance_scores = model.feature_importances_
            positions = np.arange(len(feature_names)) + idx * bar_width
            
            plt.barh(positions, importance_scores,
                    height=bar_width,
                    alpha=0.8,
                    color=colors[idx],
                    label=name)
        
        plt.yticks(np.arange(len(feature_names)) + bar_width * (len(tree_based_models) - 1) / 2,
                  feature_names)
        plt.title('Feature Importance Comparison', fontsize=14, pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def compare_models():
    try:
        # Load and prepare data
        heart_data = pd.read_csv('heart.csv')
        X = heart_data.drop(['target'], axis=1) if 'target' in heart_data.columns else heart_data
        y = heart_data['target'] if 'target' in heart_data.columns else np.zeros(len(heart_data))
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Optimize and train models
        print("Starting hyperparameter optimization...")
        best_models = optimize_hyperparameters(X_train_scaled, y_train)

        # Collect results
        results = {}
        for name, model in best_models.items():
            # Train and evaluate
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            print(f"\n{name} Test Accuracy: {accuracy:.4f}")

        # Generate detailed visualizations
        print("\nGenerating detailed visualizations...")
        plot_detailed_metrics(best_models, X_test_scaled, y_test)

        # Print comprehensive analysis
        print("\nDetailed Model Analysis")
        print("=====================")
        for name, result in results.items():
            print(f"\n{name}:")
            print("-" * len(name))
            report = result['report']
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
            print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
            print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")

        # Find best model
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\nBest Overall Model: {best_model_name}")
        
        # Save feature importance for tree-based models
        if hasattr(best_models[best_model_name], 'feature_importances_'):
            plt.figure(figsize=(12, 6), facecolor='white')
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_models[best_model_name].feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title(f'Feature Importance ({best_model_name})')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please ensure the dataset files are in the correct location and format.")

if __name__ == "__main__":
    compare_models()
