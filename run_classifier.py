# run_classifier.py
# Main script to run deadlock detection binary classification
# This script integrates with your existing Banker's Algorithm and RAG data generation

import os
import sys
import numpy as np
import pandas as pd
from deadlock_classifier import DeadlockClassifier, create_sample_data

def main():
    """
    Main function to run the complete deadlock detection classification pipeline
    """
    print("=" * 80)
    print("DEADLOCK DETECTION BINARY CLASSIFICATION SYSTEM")
    print("Banker's Algorithm vs RAG - Safe/Unsafe State Prediction")
    print("=" * 80)
    
    # Initialize classifier
    classifier = DeadlockClassifier(random_state=42)
    
    # Step 1: Load Data
    print("\n🔄 STEP 1: LOADING DATA")
    print("-" * 40)
    
    # Try to load your generated datasets
    banker_file = "deadlock_dataset_banker.csv"
    rag_file = "deadlock_dataset_rag.csv"
    
    df = None
    
    # Check if your generated files exist
    if os.path.exists(banker_file) or os.path.exists(rag_file):
        try:
            df = classifier.load_data_from_csv(
                banker_csv_path=banker_file if os.path.exists(banker_file) else None,
                rag_csv_path=rag_file if os.path.exists(rag_file) else None,
                combined=True
            )
            print("✓ Successfully loaded your generated datasets!")
            
        except Exception as e:
            print(f"✗ Error loading CSV files: {e}")
            df = None
    
    # If no files found, create sample data or prompt user
    if df is None:
        print("\n⚠️  Your generated CSV files not found.")
        print("Options:")
        print("1. Run your generate_dataset.py first to create the datasets")
        print("2. Use sample data for demonstration")
        print("3. Provide custom file paths")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            print("\n📝 Please run the following command first:")
            print("   python generate_dataset.py")
            print("\nThen run this script again.")
            return
            
        elif choice == "2":
            print("📊 Creating sample data for demonstration...")
            df = create_sample_data(n_samples=2000, num_processes=5, num_resources=3)
            print(f"✓ Created sample dataset with {len(df)} samples")
            
        elif choice == "3":
            banker_path = input("Enter path to Banker's dataset (or press Enter to skip): ").strip()
            rag_path = input("Enter path to RAG dataset (or press Enter to skip): ").strip()
            
            try:
                df = classifier.load_data_from_csv(
                    banker_csv_path=banker_path if banker_path else None,
                    rag_csv_path=rag_path if rag_path else None,
                    combined=True
                )
            except Exception as e:
                print(f"✗ Error loading custom files: {e}")
                print("Using sample data instead...")
                df = create_sample_data(n_samples=1000)
        else:
            print("Invalid choice. Using sample data...")
            df = create_sample_data(n_samples=1000)
    
    # Step 2: Feature Engineering
    print("\n🔧 STEP 2: FEATURE PREPARATION")
    print("-" * 40)
    
    X, y = classifier.prepare_features(df, num_processes=5, num_resources=3, engineer_features=True)
    
    # Step 3: Model Training and Evaluation
    print("\n🚀 STEP 3: MODEL TRAINING & EVALUATION")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = classifier.train_and_evaluate(
        X, y, 
        test_size=0.2, 
        cv_folds=5, 
        scaling='standard'
    )
    
    # Step 4: Find Best Models
    print("\n🏆 STEP 4: MODEL COMPARISON")
    print("-" * 40)
    
    top_models = classifier.find_best_models(top_k=5, metric='test_f1')
    
    # Step 5: Hyperparameter Tuning (Optional)
    print("\n⚙️ STEP 5: HYPERPARAMETER TUNING")
    print("-" * 40)
    
    tune_choice = input("Perform hyperparameter tuning? (y/n, default=y): ").strip().lower()
    
    if tune_choice != 'n':
        print("Starting hyperparameter tuning for top models...")
        
        # Select models to tune based on what's available
        models_to_tune = []
        available_models = [model[0] for model in top_models]
        
        for model in ['Random_Forest', 'XGBoost', 'Logistic_Regression', 'SVM_RBF']:
            if model in available_models:
                models_to_tune.append(model)
        
        if models_to_tune:
            tuned_models = classifier.hyperparameter_tuning(X_train, y_train, models_to_tune[:3])
            
            if tuned_models:
                print(f"✓ Tuned {len(tuned_models)} models. Re-evaluating...")
                
                # Re-evaluate with tuned models
                X_train2, X_test2, y_train2, y_test2 = classifier.train_and_evaluate(
                    X, y, test_size=0.2, cv_folds=5, scaling='standard'
                )
                
                # Find best models again
                print("\n🏆 UPDATED MODEL RANKINGS (After Tuning)")
                print("-" * 50)
                top_models = classifier.find_best_models(top_k=5)
        else:
            print("No suitable models found for tuning.")
    else:
        print("Skipping hyperparameter tuning.")
    
    # Step 6: Detailed Analysis
    print("\n📊 STEP 6: DETAILED ANALYSIS")
    print("-" * 40)
    
    # Generate classification report for best model
    classifier.generate_classification_report()
    
    # Feature importance analysis
    print("\n📈 Feature Importance Analysis:")
    feature_importance = classifier.get_feature_importance(top_k=20)
    
    # Step 7: Visualization
    print("\n📊 STEP 7: VISUALIZATION")
    print("-" * 40)
    
    plot_choice = input("Generate performance plots? (y/n, default=y): ").strip().lower()
    
    if plot_choice != 'n':
        try:
            classifier.plot_results(save_path="deadlock_classification_results.png")
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
            print("This might be due to missing matplotlib or display issues.")
    
    # Step 8: Model Saving
    print("\n💾 STEP 8: MODEL SAVING")
    print("-" * 40)
    
    save_choice = input("Save the best model? (y/n, default=y): ").strip().lower()
    
    if save_choice != 'n':
        classifier.save_model()
        results_df = classifier.export_results()
        print("\n📊 Results Summary:")
        print(results_df[['Model', 'F1_Score', 'Balanced_Accuracy', 'Precision', 'Recall']].head())
    
    # Step 9: Interactive Prediction (Fixed)
    print("\n🎯 STEP 9: INTERACTIVE PREDICTION")
    print("-" * 40)
    
    interactive_choice = input("Try interactive prediction? (y/n, default=y): ").strip().lower()
    
    if interactive_choice != 'n':
        demo_interactive_prediction(classifier)
    
    # Final Summary and Recommendations
    print("\n" + "=" * 80)
    print("🎉 ENHANCED CLASSIFICATION COMPLETED!")
    print("=" * 80)
    
    if classifier.best_model:
        best = classifier.best_model['results']
        print(f"🏆 Best Model: {classifier.best_model['name']}")
        print(f"   📈 F1-Score: {best['test_f1']:.4f}")
        print(f"   📈 Accuracy: {best['test_accuracy']:.4f}")
        print(f"   📈 Balanced Accuracy: {best['test_balanced_accuracy']:.4f}")
        print(f"   📈 Precision: {best['test_precision']:.4f}")
        print(f"   📈 Recall: {best['test_recall']:.4f}")
        if best.get('test_auc'):
            print(f"   📈 AUC: {best['test_auc']:.4f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if best['test_f1'] < 0.5:
        print(f"   🔧 Performance needs improvement:")
        print(f"      - Check RAG algorithm for 0 safe states issue")
        print(f"      - Install imbalanced-learn: pip install imbalanced-learn")
        print(f"      - Consider ensemble methods")
    elif best['test_f1'] < 0.7:
        print(f"   ✅ Moderate performance. For safety-critical systems:")
        print(f"      - Monitor false positive rate closely")
        print(f"      - Consider ensemble methods")
        print(f"      - Validate on diverse test cases")
    else:
        print(f"   🌟 Good performance!")
        print(f"      - Ready for further validation")
        print(f"      - Test with real-world scenarios")
    
    # Show generated files
    print(f"\n📁 Generated Files:")
    for file in os.listdir('.'):
        if 'deadlock' in file and (file.endswith('.csv') or file.endswith('.joblib')):
            print(f"   - {file}")
    
    return classifier

def demo_interactive_prediction(classifier):
    """
    Enhanced interactive demonstration of single prediction
    """
    print("\n🎯 INTERACTIVE PREDICTION DEMO")
    print("Testing the enhanced classifier with realistic scenarios")
    print("-" * 60)
    
    try:
        # Test Case 1: Likely Safe State
        print("🧪 TEST CASE 1: Likely Safe State")
        print("System with sufficient resources for all processes")
        
        available = [3, 3, 2]
        allocation = [
            [0, 1, 0],  # P0: minimal allocation
            [2, 0, 0],  # P1: some allocation
            [3, 0, 2],  # P2: moderate allocation
            [2, 1, 1],  # P3: moderate allocation
            [0, 0, 2]   # P4: minimal allocation
        ]
        need = [
            [1, 0, 0],  # P0: low remaining need
            [0, 1, 2],  # P1: low remaining need
            [0, 0, 0],  # P2: no remaining need
            [0, 0, 0],  # P3: no remaining need
            [0, 2, 0]   # P4: low remaining need
        ]
        
        print(f"Available: {available}")
        print(f"Allocation: {allocation}")
        print(f"Need: {need}")
        
        pred1, prob1 = classifier.predict_single(available, allocation, need)
        
        # Test Case 2: Likely Unsafe State
        print(f"\n🧪 TEST CASE 2: Likely Unsafe State")
        print(f"System with resource contention and potential deadlock")
        
        available = [0, 0, 1]  # Very limited resources
        allocation = [
            [2, 1, 3],  # P0: holding many resources
            [1, 2, 0],  # P1: holding some resources
            [2, 0, 1],  # P2: holding some resources
            [1, 1, 2],  # P3: holding some resources
            [0, 1, 0]   # P4: minimal resources
        ]
        need = [
            [0, 2, 0],  # P0: needs more resources
            [2, 0, 3],  # P1: needs many resources
            [1, 3, 2],  # P2: needs many resources
            [0, 1, 0],  # P3: needs some resources
            [3, 1, 2]   # P4: needs many resources
        ]
        
        print(f"Available: {available}")
        print(f"Allocation: {allocation}")
        print(f"Need: {need}")
        
        pred2, prob2 = classifier.predict_single(available, allocation, need)
        
        # Summary
        print(f"\n📊 PREDICTION SUMMARY:")
        print("-" * 30)
        print(f"Test Case 1 (Expected Safe): {'✅ Correct' if pred1 == 1 else '❌ Incorrect'}")
        print(f"Test Case 2 (Expected Unsafe): {'✅ Correct' if pred2 == 0 else '❌ Incorrect'}")
        
        if pred1 == 1 and pred2 == 0:
            print(f"🎉 Model correctly distinguished between safe and unsafe states!")
        elif pred1 == pred2:
            print(f"⚠️ Model gave same prediction for both cases - may need more training")
        else:
            print(f"🤔 Mixed results - model may be learning but needs refinement")
            
    except Exception as e:
        print(f"Error in interactive prediction: {e}")
        print("This might indicate issues with feature engineering or model training.")

def create_sample_data(n_samples=2000, num_processes=5, num_resources=3, random_state=42):
    """
    Create more realistic sample data with better class balance
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
    
    # Generate more realistic data with better balance
    data = []
    labels = []
    
    target_safe_ratio = 0.3  # 30% safe states
    target_safe_count = int(n_samples * target_safe_ratio)
    
    for i in range(n_samples):
        # Decide if this should be safe or unsafe
        should_be_safe = len([l for l in labels if l == 1]) < target_safe_count
        
        if should_be_safe:
            # Generate a likely safe state
            available = np.random.randint(2, 6, num_resources)  # More resources available
            max_per_process = 3  # Lower resource requirements
        else:
            # Generate a likely unsafe state
            available = np.random.randint(0, 3, num_resources)  # Fewer resources
            max_per_process = 6  # Higher resource requirements
        
        sample = []
        
        # Available resources
        sample.extend(available)
        
        # Allocation and need matrices
        total_allocated = np.zeros(num_resources)
        
        for p in range(num_processes):
            # Generate allocation
            max_alloc = np.minimum(available + total_allocated, max_per_process)
            allocation = np.random.randint(0, max_alloc + 1, num_resources)
            sample.extend(allocation)
            total_allocated += allocation
        
        for p in range(num_processes):
            # Generate need (remaining requirements)
            if should_be_safe:
                need = np.random.randint(0, 3, num_resources)  # Lower needs for safe states
            else:
                need = np.random.randint(0, max_per_process, num_resources)  # Higher needs
            sample.extend(need)
        
        data.append(sample)
        labels.append(1 if should_be_safe else 0)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['label'] = labels
    
    safe_count = np.sum(labels)
    print(f"Created sample dataset: {len(df)} samples")
    print(f"  - Safe states: {safe_count} ({safe_count/len(labels)*100:.1f}%)")
    print(f"  - Unsafe states: {len(labels)-safe_count} ({(len(labels)-safe_count)/len(labels)*100:.1f}%)")
    
    return df

def check_dependencies():
    """
    Enhanced dependency checking with recommendations
    """
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib'
    ]
    
    optional_packages = [
        ('imbalanced-learn', 'imblearn'),
        ('xgboost', 'xgboost'),  
        ('lightgbm', 'lightgbm')
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package_name, import_name in optional_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_optional.append(package_name)
    
    if missing_required:
        print("❌ Missing required packages:")
        for package in missing_required:
            print(f"   - {package}")
        print(f"\n📦 Install with: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print("⚠️ Missing optional packages (recommended for better performance):")
        for package in missing_optional:
            print(f"   - {package}")
        print(f"\n📦 Install with: pip install {' '.join(missing_optional)}")
    
    return True

if __name__ == "__main__":
    print("Checking dependencies...")
    
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies available!")
    
    try:
        classifier = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your data files and try again.")