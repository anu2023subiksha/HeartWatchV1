import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from HeartUlcerModel import HealthPredictionModel
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_basic_metrics():
    # Initialize and train the models
    health_model = HealthPredictionModel()
    health_model.prepare_combined_dataset()
    health_model.preprocess_data()
    health_model.train_model()

    # Get predictions and probabilities
    y_pred = health_model.model.predict(health_model.X_test)
    y_true = health_model.y_test
    y_prob = health_model.model.predict_proba(health_model.X_test)

    # 1. Basic Metrics Visualization
    plt.figure(figsize=(15, 10))

    # Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Feature Importance
    plt.subplot(2, 2, 2)
    feature_importance = pd.DataFrame({
        'feature': health_model.features,
        'importance': health_model.model.feature_importances_
    }).sort_values('importance', ascending=True)
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')

    # Class Distribution
    plt.subplot(2, 2, 3)
    class_dist = pd.Series(y_true).value_counts()
    plt.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%')
    plt.title('Class Distribution')

    # Performance Metrics
    plt.subplot(2, 2, 4)
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    plt.axis('off')
    plt.table(cellText=np.round(metrics_df, 3).values,
             rowLabels=metrics_df.index,
             colLabels=metrics_df.columns,
             cellLoc='center',
             loc='center')
    plt.title('Classification Report')

    plt.tight_layout()
    plt.savefig('basic_metrics.png')
    plt.close()

def plot_feature_correlations():
    # Initialize model and get data
    health_model = HealthPredictionModel()
    health_model.prepare_combined_dataset()
    
    # Create correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = health_model.X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()

def plot_feature_distributions():
    # Initialize model and get data
    health_model = HealthPredictionModel()
    health_model.prepare_combined_dataset()
    
    # Plot distributions for each feature
    features = health_model.features
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data=health_model.X, x=feature, hue=health_model.y, multiple="stack")
        plt.title(f'{feature} Distribution by Class')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def plot_roc_curves():
    # Initialize and train the models
    health_model = HealthPredictionModel()
    health_model.prepare_combined_dataset()
    health_model.preprocess_data()
    health_model.train_model()

    # Get predictions and probabilities
    y_prob = health_model.model.predict_proba(health_model.X_test)
    
    # Prepare data for ROC curves
    classes = sorted(set(health_model.y))
    y_test_bin = label_binarize(health_model.y_test, classes=classes)
    n_classes = len(classes)

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green'])
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'ROC curve of {classes[i]} (AUC = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.close()

def plot_precision_recall_curves():
    # Initialize and train the models
    health_model = HealthPredictionModel()
    health_model.prepare_combined_dataset()
    health_model.preprocess_data()
    health_model.train_model()

    # Get predictions and probabilities
    y_prob = health_model.model.predict_proba(health_model.X_test)
    
    # Prepare data for PR curves
    classes = sorted(set(health_model.y))
    y_test_bin = label_binarize(health_model.y_test, classes=classes)
    n_classes = len(classes)

    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green'])
    
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                label=f'Precision-Recall curve for {classes[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png')
    plt.close()

def plot_model_metrics(models, X_test, y_test):
    # Calculate metrics for each model
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate various scores
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        z_score = np.abs(stats.zscore(y_pred)).mean()
        correlation = stats.pointbiserialr(y_test, y_pred)[0]
        
        # Calculate distribution value (entropy)
        _, counts = np.unique(y_pred, return_counts=True)
        distribution = stats.entropy(counts)
        
        metrics[name] = {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Z-Score': z_score,
            'Correlation': correlation,
            'Distribution': distribution
        }
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics).T
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Line plot with points for main metrics
    metrics_to_plot = ['Accuracy', 'F1 Score', 'Correlation']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for metric, color in zip(metrics_to_plot, colors):
        values = df[metric]
        ax1.plot(values.index, values.values, marker='o', label=metric, 
                linewidth=2, markersize=10, color=color)
        
        # Add value labels
        for x, y in zip(values.index, values.values):
            ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    ax1.set_title('Model Performance Comparison', fontsize=14, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.1])
    ax1.set_xticklabels(df.index, rotation=45)
    
    # Plot 2: Distribution and Z-Score comparison
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df['Distribution'], width, label='Distribution',
                    color='#3498db', alpha=0.7)
    bars2 = ax2.bar(x + width/2, df['Z-Score'], width, label='Z-Score',
                    color='#e74c3c', alpha=0.7)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    ax2.set_title('Distribution and Z-Score Comparison', fontsize=14, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models.keys(), rotation=45)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create heatmap of all metrics
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='RdYlGn', fmt='.3f',
                center=0.5, vmin=0, vmax=1)
    plt.title('All Metrics Comparison Heatmap', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('model_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Load and prepare data
        print("Loading data...")
        heart_data = pd.read_csv('heart.csv')
        X = heart_data.drop(['target'], axis=1)
        y = heart_data['target']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models with best parameters
        print("\nInitializing models...")
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=2, random_state=42
            ),
            'Gradient Boost': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'SVM': SVC(
                C=10, gamma='scale', kernel='rbf', random_state=42, probability=True
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, min_samples_split=2, criterion='gini', random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=150, learning_rate=1.0, algorithm='SAMME', random_state=42
            )
        }
        
        # Train models
        print("Training models...")
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_model_metrics(models, X_test_scaled, y_test)
        
        print("\nVisualization completed! Check:")
        print("1. model_metrics_comparison.png")
        print("2. model_metrics_heatmap.png")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please ensure the dataset files are in the correct location and format.")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    plot_basic_metrics()
    plot_feature_correlations()
    plot_feature_distributions()
    plot_roc_curves()
    plot_precision_recall_curves()
    main()
