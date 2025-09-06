# deadlock_classifier.py
# Binary Classification for Deadlock Detection: Banker's Algorithm vs RAG
# Specialized classifier for your deadlock detection project

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, classification_report, confusion_matrix, 
                           roc_curve, precision_recall_curve)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            VotingClassifier, BaggingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallback
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

class DeadlockClassifier:
    """
    Specialized binary classifier for deadlock detection using Banker's Algorithm and RAG data
    
    Features:
    - Loads data from your generated CSV files
    - Domain-specific feature engineering for deadlock detection
    - Multiple ML algorithms optimized for safe/unsafe classification
    - Comprehensive evaluation and model comparison
    - Feature importance analysis
    - Model saving/loading for deployment
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the DeadlockClassifier
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_names = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data_from_csv(self, banker_csv_path=None, rag_csv_path=None, combined=True):
        """
        Load your generated datasets from CSV files
        
        Args:
            banker_csv_path (str): Path to banker's algorithm dataset
            rag_csv_path (str): Path to RAG dataset
            combined (bool): Whether to combine both datasets
            
        Returns:
            pandas.DataFrame: Loaded dataset(s)
        """
        data = []
        
        if banker_csv_path:
            try:
                banker_df = pd.read_csv(banker_csv_path)
                banker_df['algorithm'] = 'banker'
                data.append(banker_df)
                print(f"✓ Loaded Banker's data: {len(banker_df)} samples")
                print(f"  - Safe states: {len(banker_df[banker_df['label']==1])}")
                print(f"  - Unsafe states: {len(banker_df[banker_df['label']==0])}")
            except FileNotFoundError:
                print(f"✗ File not found: {banker_csv_path}")
        
        if rag_csv_path:
            try:
                rag_df = pd.read_csv(rag_csv_path)
                rag_df['algorithm'] = 'rag'
                data.append(rag_df)
                print(f"✓ Loaded RAG data: {len(rag_df)} samples")
                print(f"  - Safe states: {len(rag_df[rag_df['label']==1])}")
                print(f"  - Unsafe states: {len(rag_df[rag_df['label']==0])}")
            except FileNotFoundError:
                print(f"✗ File not found: {rag_csv_path}")
        
        if not data:
            raise ValueError("No data files could be loaded. Please check file paths.")
        
        if combined and len(data) > 1:
            combined_df = pd.concat(data, ignore_index=True)
            print(f"✓ Combined dataset: {len(combined_df)} total samples")
            print(f"  - Total Safe states: {len(combined_df[combined_df['label']==1])}")
            print(f"  - Total Unsafe states: {len(combined_df[combined_df['label']==0])}")
            return combined_df
        else:
            return data[0] if len(data) == 1 else data
    
    def prepare_features(self, df, num_processes=5, num_resources=3, engineer_features=True):
        """
        Extract and prepare features from your deadlock detection data
        
        Args:
            df (pandas.DataFrame): Input dataframe
            num_processes (int): Number of processes in the system
            num_resources (int): Number of resource types
            engineer_features (bool): Whether to create additional engineered features
            
        Returns:
            tuple: (X, y) - features and labels
        """
        print(f"\nPreparing features...")
        print(f"  - Processes: {num_processes}")
        print(f"  - Resources: {num_resources}")
        print(f"  - Engineer features: {engineer_features}")
        
        # Feature columns (excluding label and algorithm if present)
        feature_cols = [col for col in df.columns if col not in ['label', 'algorithm']]
        
        X = df[feature_cols].values
        y = df['label'].values
        
        # Store original feature names
        self.feature_names = feature_cols.copy()
        
        # Create additional engineered features specific to deadlock detection
        if engineer_features:
            X_engineered = self._engineer_deadlock_features(df, num_processes, num_resources)
            
            if X_engineered is not None:
                X = np.column_stack([X, X_engineered])
                
                # Update feature names
                engineered_names = self._get_engineered_feature_names(num_processes, num_resources)
                self.feature_names = feature_cols + engineered_names
                print(f"  - Added {len(engineered_names)} engineered features")
        
        print(f"  - Total features: {len(self.feature_names)}")
        print(f"  - Total samples: {len(X)}")
        
        return X, y
    
    def _engineer_deadlock_features(self, df, num_processes=5, num_resources=3):
        """
        Create domain-specific features for deadlock detection
        """
        engineered_features = []
        
        try:
            # Extract available resources
            available_cols = [f'Available_R{r}' for r in range(num_resources)]
            available_matrix = df[available_cols].values
            
            # Extract allocation matrix
            alloc_features = []
            for p in range(num_processes):
                for r in range(num_resources):
                    alloc_features.append(f'P{p}_Alloc_R{r}')
            allocation_matrix = df[alloc_features].values.reshape(-1, num_processes, num_resources)
            
            # Extract need matrix
            need_features = []
            for p in range(num_processes):
                for r in range(num_resources):
                    need_features.append(f'P{p}_Need_R{r}')
            need_matrix = df[need_features].values.reshape(-1, num_processes, num_resources)
            
            # Feature Engineering
            for i in range(len(df)):
                sample_features = []
                
                # 1. Total resources in system
                total_resources = available_matrix[i] + np.sum(allocation_matrix[i], axis=0)
                sample_features.extend(total_resources)
                
                # 2. Resource utilization ratio
                utilization = np.sum(allocation_matrix[i], axis=0) / (total_resources + 1e-8)
                sample_features.extend(utilization)
                
                # 3. Process satisfaction ratio (how much of max need is satisfied)
                max_need = allocation_matrix[i] + need_matrix[i]
                satisfaction = np.mean(allocation_matrix[i] / (max_need + 1e-8), axis=1)
                sample_features.extend(satisfaction)
                
                # 4. Critical processes (high need relative to available)
                critical_processes = np.sum(need_matrix[i] > available_matrix[i].reshape(1, -1), axis=1)
                sample_features.extend(critical_processes)
                
                # 5. Resource bottlenecks
                bottleneck_resources = (available_matrix[i] == 0).astype(int)
                sample_features.extend(bottleneck_resources)
                
                # 6. Total system demand
                total_demand = np.sum(need_matrix[i])
                total_available = np.sum(available_matrix[i])
                demand_ratio = total_demand / (total_available + 1e-8)
                sample_features.append(demand_ratio)
                
                # 7. Process competition (processes competing for same resources)
                competition_score = 0
                for r in range(num_resources):
                    competing_processes = np.sum(need_matrix[i][:, r] > 0)
                    if competing_processes > 1 and available_matrix[i][r] < np.max(need_matrix[i][:, r]):
                        competition_score += competing_processes
                sample_features.append(competition_score)
                
                engineered_features.append(sample_features)
            
            return np.array(engineered_features)
            
        except Exception as e:
            print(f"Warning: Feature engineering failed: {e}")
            return None
    
    def _get_engineered_feature_names(self, num_processes=5, num_resources=3):
        """
        Get names for engineered features
        """
        names = []
        
        # Total resources
        names.extend([f'Total_R{r}' for r in range(num_resources)])
        
        # Utilization ratios
        names.extend([f'Util_R{r}' for r in range(num_resources)])
        
        # Process satisfaction
        names.extend([f'Satisfaction_P{p}' for p in range(num_processes)])
        
        # Critical processes
        names.extend([f'Critical_P{p}' for p in range(num_processes)])
        
        # Bottleneck resources
        names.extend([f'Bottleneck_R{r}' for r in range(num_resources)])
        
        # System metrics
        names.extend(['System_Demand_Ratio', 'Competition_Score'])
        
        return names
    
    def prepare_models(self):
        """
        Initialize models optimized for deadlock detection
        """
        
        self.models = {
            # High-performance ensemble methods (recommended for your use case)
            'Random_Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=self.random_state
            ),
            
            # Linear models (fast and interpretable)
            'Logistic_Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight='balanced',
                solver='liblinear'
            ),
            
            'Ridge_Classifier': RidgeClassifier(
                alpha=1.0,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            # Tree-based models
            'Decision_Tree': DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            # Instance-based learning
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='euclidean'
            ),
            
            # Probabilistic models
            'Naive_Bayes': GaussianNB(),
            
            'LDA': LinearDiscriminantAnalysis(),
            
            # Support Vector Machines
            'SVM_Linear': SVC(
                kernel='linear',
                C=1.0,
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            ),
            
            'SVM_RBF': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            ),
            
            # Neural Network
            'Neural_Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            ),
            
            # Additional ensemble methods
            'Ada_Boost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=self.random_state
            ),
            
            'Bagging': BaggingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # Add gradient boosting models if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
            )
        
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = LGBMClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            )
        
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostClassifier(
                iterations=150,
                learning_rate=0.1,
                depth=8,
                random_seed=self.random_state,
                verbose=False
            )
        
        print(f"✓ Initialized {len(self.models)} models")
    
    def create_ensemble_models(self):
        """
        Create ensemble models combining the best performers
        """
        base_models = []
        
        if 'Random_Forest' in self.models:
            base_models.append(('rf', self.models['Random_Forest']))
        if 'XGBoost' in self.models:
            base_models.append(('xgb', self.models['XGBoost']))
        if 'Logistic_Regression' in self.models:
            base_models.append(('lr', self.models['Logistic_Regression']))
        if 'SVM_RBF' in self.models:
            base_models.append(('svm', self.models['SVM_RBF']))
        
        if len(base_models) >= 3:
            # Voting classifiers
            self.models['Voting_Hard'] = VotingClassifier(
                estimators=base_models[:3], voting='hard'
            )
            
            self.models['Voting_Soft'] = VotingClassifier(
                estimators=base_models[:3], voting='soft'
            )
            
            print(f"✓ Added ensemble models using top {min(3, len(base_models))} base models")
    
    def train_and_evaluate(self, X, y, test_size=0.2, cv_folds=5, scaling='standard'):
        """
        Complete training and evaluation pipeline
        """
        print("=" * 70)
        print("DEADLOCK DETECTION - BINARY CLASSIFICATION PIPELINE")
        print("=" * 70)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Dataset Information:")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Testing samples: {len(self.X_test)}")
        print(f"  - Total features: {X.shape[1]}")
        print(f"  - Class distribution:")
        print(f"    • Safe states (1): {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
        print(f"    • Unsafe states (0): {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
        
        # Scaling
        if scaling == 'standard':
            self.scaler = StandardScaler()
        elif scaling == 'robust':
            self.scaler = RobustScaler()
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        
        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            print(f"  - Applied {scaling} scaling")
        else:
            X_train_scaled, X_test_scaled = self.X_train, self.X_test
            print(f"  - No scaling applied")
        
        # Initialize models
        self.prepare_models()
        self.create_ensemble_models()
        
        # Evaluation
        self.results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        print(f"\nTraining and evaluating {len(self.models)} models...")
        print("-" * 70)
        
        for i, (name, model) in enumerate(self.models.items(), 1):
            try:
                print(f"[{i:2d}/{len(self.models)}] Training {name}...", end=' ')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring='f1')
                
                # Train model
                model.fit(X_train_scaled, self.y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test_scaled)
                
                # Calculate metrics
                self.results[name] = {
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'test_accuracy': accuracy_score(self.y_test, y_pred),
                    'test_precision': precision_score(self.y_test, y_pred),
                    'test_recall': recall_score(self.y_test, y_pred),
                    'test_f1': f1_score(self.y_test, y_pred),
                    'test_auc': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None,
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Print results
                print(f"F1: {self.results[name]['test_f1']:.4f}, Acc: {self.results[name]['test_accuracy']:.4f} ✓")
                
            except Exception as e:
                print(f"Error: {str(e)} ✗")
        
        print(f"\n✓ Training completed for {len(self.results)} models")
        return X_train_scaled, X_test_scaled, self.y_train, self.y_test
    
    def find_best_models(self, top_k=5, metric='test_f1'):
        """
        Find the best performing models
        """
        if not self.results:
            print("No results available. Run train_and_evaluate first.")
            return None
        
        # Sort by specified metric
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1].get(metric, 0), reverse=True)
        
        print(f"\nTOP {top_k} MODELS (sorted by {metric}):")
        print("=" * 70)
        
        for i, (name, results) in enumerate(sorted_models[:top_k]):
            print(f"{i+1:2d}. {name:<20}")
            print(f"    F1-Score:  {results['test_f1']:.4f}")
            print(f"    Accuracy:  {results['test_accuracy']:.4f}")
            print(f"    Precision: {results['test_precision']:.4f}")
            print(f"    Recall:    {results['test_recall']:.4f}")
            if results.get('test_auc'):
                print(f"    AUC:       {results['test_auc']:.4f}")
            print()
        
        self.best_model = {
            'name': sorted_models[0][0],
            'results': sorted_models[0][1]
        }
        
        print(f"🏆 Best Model: {self.best_model['name']} (F1: {self.best_model['results']['test_f1']:.4f})")
        
        return sorted_models[:top_k]
    
    def hyperparameter_tuning(self, X_train, y_train, models_to_tune=None):
        """
        Perform hyperparameter tuning for selected models
        """
        if models_to_tune is None:
            models_to_tune = ['Random_Forest', 'XGBoost', 'Logistic_Regression']
        
        # Filter available models
        models_to_tune = [m for m in models_to_tune if m in self.models]
        
        param_grids = {
            'Random_Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 1.0]
            } if XGBOOST_AVAILABLE else {},
            'Logistic_Regression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            },
            'SVM_RBF': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        }
        
        tuned_models = {}
        
        print(f"\nHYPERPARAMETER TUNING:")
        print("=" * 40)
        
        for model_name in models_to_tune:
            if model_name in param_grids and param_grids[model_name]:
                print(f"Tuning {model_name}...")
                
                grid_search = GridSearchCV(
                    self.models[model_name],
                    param_grids[model_name],
                    cv=3,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                try:
                    grid_search.fit(X_train, y_train)
                    tuned_models[f'{model_name}_Tuned'] = grid_search.best_estimator_
                    
                    print(f"  ✓ Best F1: {grid_search.best_score_:.4f}")
                    print(f"  ✓ Best params: {grid_search.best_params_}")
                    print()
                except Exception as e:
                    print(f"  ✗ Error tuning {model_name}: {str(e)}")
        
        # Add tuned models to main models
        self.models.update(tuned_models)
        print(f"✓ Added {len(tuned_models)} tuned models")
        
        return tuned_models
    
    def plot_results(self, save_path=None, figsize=(16, 12)):
        """
        Visualize model performance
        """
        if not self.results:
            print("No results to plot. Run train_and_evaluate first.")
            return
        
        # Prepare data
        models = list(self.results.keys())
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Deadlock Detection Model Performance Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [self.results[model].get(metric, 0) for model in models]
            
            bars = ax.bar(range(len(models)), values, color=colors)
            ax.set_title(metric.replace('test_', '').replace('_', ' ').title(), 
                        fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def get_feature_importance(self, model_name=None, top_k=15):
        """
        Get feature importance for tree-based models
        """
        if model_name is None and self.best_model:
            model_name = self.best_model['name']
        
        if model_name not in self.results:
            print(f"Model {model_name} not found.")
            return None
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            if self.feature_names:
                feature_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                print(f"\nTOP {top_k} IMPORTANT FEATURES ({model_name}):")
                print("=" * 60)
                for i, (_, row) in enumerate(feature_df.head(top_k).iterrows()):
                    print(f"{i+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
                print()
                
                return feature_df
            else:
                return importance
        else:
            print(f"Model {model_name} doesn't support feature importance.")
            return None
    
    def generate_classification_report(self, model_name=None):
        """
        Generate detailed classification report
        """
        if model_name is None and self.best_model:
            model_name = self.best_model['name']
        
        if model_name not in self.results:
            print(f"Model {model_name} not found.")
            return
        
        y_pred = self.results[model_name]['predictions']
        
        print(f"\nCLASSIFICATION REPORT - {model_name}")
        print("=" * 50)
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Unsafe (0)', 'Safe (1)'],
                                  digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"CONFUSION MATRIX:")
        print(f"                Predicted")
        print(f"                0    1")
        print(f"Actual    0   {cm[0,0]:4d} {cm[0,1]:4d}")
        print(f"          1   {cm[1,0]:4d} {cm[1,1]:4d}")
        print()
    
    def save_model(self, model_name=None, filepath=None):
        """
        Save the best model for deployment
        """
        if model_name is None and self.best_model:
            model_name = self.best_model['name']
        
        if filepath is None:
            filepath = f"deadlock_classifier_{model_name.lower().replace(' ', '_')}.joblib"
        
        if model_name in self.results:
            model_data = {
                'model': self.results[model_name]['model'],
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'results': self.results[model_name],
                'model_name': model_name,
                'random_state': self.random_state
            }
            
            joblib.dump(model_data, filepath)
            print(f"✓ Model '{model_name}' saved to {filepath}")
            print(f"  - F1 Score: {self.results[model_name]['test_f1']:.4f}")
            print(f"  - Accuracy: {self.results[model_name]['test_accuracy']:.4f}")
        else:
            print(f"✗ Model {model_name} not found.")
    
    def load_model(self, filepath):
        """
        Load a previously saved model
        """
        try:
            model_data = joblib.load(filepath)
            
            print(f"✓ Model loaded from {filepath}")
            print(f"  - Model: {model_data['model_name']}")
            print(f"  - F1 Score: {model_data['results']['test_f1']:.4f}")
            print(f"  - Accuracy: {model_data['results']['test_accuracy']:.4f}")
            
            return model_data
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return None
    
    def predict(self, X, model_name=None):
        """
        Make predictions using trained model
        """
        if model_name is None and self.best_model:
            model_name = self.best_model['name']
        
        if model_name not in self.results:
            print(f"Model {model_name} not found. Train the model first.")
            return None
        
        model = self.results[model_name]['model']
        
        # Apply scaling if used during training
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = None
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def predict_single(self, available_resources, allocation_matrix, need_matrix, model_name=None):
        """
        Predict safety for a single system state (fixed to match training features)
        
        Args:
            available_resources: List of available resources [R0, R1, R2]
            allocation_matrix: 2D list of current allocations [[P0_R0, P0_R1, P0_R2], ...]
            need_matrix: 2D list of resource needs [[P0_R0, P0_R1, P0_R2], ...]
            model_name: Name of model to use (default: best model)
            
        Returns:
            tuple: (prediction, probability) - (1=safe, 0=unsafe), confidence
        """
        if model_name is None and self.best_model:
            model_name = self.best_model['name']
        
        if model_name not in self.results:
            print(f"Model {model_name} not found. Available models: {list(self.results.keys())}")
            return None, None
        
        # Create DataFrame exactly like training data
        sample_data = {}
        
        # Available resources
        for r in range(len(available_resources)):
            sample_data[f'Available_R{r}'] = [available_resources[r]]
        
        # Allocation matrix (flattened)
        for p in range(len(allocation_matrix)):
            for r in range(len(allocation_matrix[0])):
                sample_data[f'P{p}_Alloc_R{r}'] = [allocation_matrix[p][r]]
        
        # Need matrix (flattened)
        for p in range(len(need_matrix)):
            for r in range(len(need_matrix[0])):
                sample_data[f'P{p}_Need_R{r}'] = [need_matrix[p][r]]
        
        # Create DataFrame with dummy label
        df_single = pd.DataFrame(sample_data)
        df_single['label'] = [0]  # Dummy label
        
        # Use the same feature preparation process as training
        X_single, _ = self.prepare_features(df_single, 
                                          num_processes=len(allocation_matrix), 
                                          num_resources=len(available_resources),
                                          engineer_features=True)
        
        # Apply same scaling as training
        if self.scaler:
            X_single_scaled = self.scaler.transform(X_single)
        else:
            X_single_scaled = X_single
        
        # Make prediction
        model = self.results[model_name]['model']
        prediction = model.predict(X_single_scaled)[0]
        probability = None
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X_single_scaled)[0, 1]
        
        status = "🟢 SAFE" if prediction == 1 else "🔴 UNSAFE"
        confidence = f"{probability:.4f}" if probability is not None else "N/A"
        
        print(f"\n🎯 PREDICTION: {status}")
        print(f"🎯 CONFIDENCE: {confidence}")
        
        if prediction == 0:
            print("⚠️  This system state may lead to deadlock!")
        else:
            print("✅ This system state appears to be safe.")
        
        return prediction, probability
    
    def compare_algorithms(self):
        """
        Compare performance between Banker's and RAG algorithms if both datasets were loaded
        """
        # This would require tracking which samples came from which algorithm
        # Implementation depends on how you want to structure the comparison
        print("Algorithm comparison feature - to be implemented based on your specific needs")
    
    def export_results(self, filepath=None):
        """
        Export comprehensive results to CSV for further analysis
        """
        if not self.results:
            print("No results to export.")
            return None
        
        if filepath is None:
            filepath = "deadlock_classification_results.csv"
        
        # Prepare results data with enhanced metrics
        results_data = []
        for model_name, results in self.results.items():
            results_data.append({
                'Model': model_name,
                'F1_Score': results['test_f1'],
                'Accuracy': results['test_accuracy'],
                'Balanced_Accuracy': results['test_balanced_accuracy'],
                'Precision': results['test_precision'],
                'Recall': results['test_recall'],
                'AUC': results.get('test_auc', 'N/A'),
                'CV_F1_Mean': results['cv_f1_mean'],
                'CV_F1_Std': results['cv_f1_std']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('F1_Score', ascending=False)
        results_df.to_csv(filepath, index=False)
        print(f"✓ Results exported to {filepath}")
        
        return results_df

# Utility functions for integration with your existing code

def create_sample_data(n_samples=1000, num_processes=5, num_resources=3, random_state=42):
    """
    Create sample data in your format for testing purposes
    """
    np.random.seed(random_state)
    
    # Create feature names matching your format
    feature_names = []
    
    # Available resources
    for r in range(num_resources):
        feature_names.append(f'Available_R{r}')
    
    # Allocation matrix
    for p in range(num_processes):
        for r in range(num_resources):
            feature_names.append(f'P{p}_Alloc_R{r}')
    
    # Need matrix
    for p in range(num_processes):
        for r in range(num_resources):
            feature_names.append(f'P{p}_Need_R{r}')
    
    # Generate random data
    data = []
    for _ in range(n_samples):
        sample = []
        
        # Available resources (0-10)
        available = np.random.randint(0, 11, num_resources)
        sample.extend(available)
        
        # Allocation matrix (0-5 per process-resource pair)
        allocation = np.random.randint(0, 6, num_processes * num_resources)
        sample.extend(allocation)
        
        # Need matrix (0-7 per process-resource pair)
        need = np.random.randint(0, 8, num_processes * num_resources)
        sample.extend(need)
        
        data.append(sample)
    
    # Generate random labels (more realistic distribution)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])  # 70% safe, 30% unsafe
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['label'] = labels
    
    return df

if __name__ == "__main__":
    print("DeadlockClassifier module loaded successfully!")
    print("Use this module by importing: from deadlock_classifier import DeadlockClassifier")
    print("For a complete example, run: python run_classifier.py")