import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

class ModelTrainer:
    def __init__(self, processed_data):
        """
        Initialize the model trainer with processed data.
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary containing processed data components
        """
        self.processed_data = processed_data
        self.models = None
        self.cv_results = None
    
    def train_model(self, model_type):
        """
        Train a logistic regression model for the specified prediction task.
        
        Parameters:
        -----------
        model_type : int
            0 for Spread, 1 for Money Line, 2 for Over/Under
        
        Returns:
        --------
        tuple: (model, cross_validation_scores)
        """
        X_train = self.processed_data['X_train']
        y_train = self.processed_data['y_train'][model_type]
        
        # Check if we have at least 2 classes in the target
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            # Create a dummy model that always predicts the only class
            class DummyClassifier:
                def __init__(self, constant_class):
                    self.constant_class = constant_class
                    # Add dummy coefficients for feature importance
                    self.coef_ = np.zeros((1, X_train.shape[1]))
                    self.intercept_ = np.array([0.0])
                    
                def predict(self, X):
                    return np.full(X.shape[0], self.constant_class)
                    
                def predict_proba(self, X):
                    probs = np.zeros((X.shape[0], 2))
                    idx = 1 if self.constant_class == 1 else 0
                    probs[:, idx] = 1.0
                    return probs
            
            # Create dummy model and scores
            model = DummyClassifier(unique_classes[0])
            cv_scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Perfect accuracy since only one class
            
            return model, cv_scores
        
        # Initialize model with L2 regularization and balanced class weights
        model = LogisticRegression(
            C=1.0,  # Inverse of regularization strength
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Train model with try-except to handle edge cases
        try:
            model.fit(X_train, y_train)
            
            # Perform cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
            except Exception as e:
                # If cross-validation fails, use a fallback
                print(f"Cross-validation failed: {e}")
                cv_scores = np.array([model.score(X_train, y_train)] * 5)
                
        except Exception as e:
            print(f"Model training failed: {e}")
            # Create a fallback model that predicts the most common class
            most_common = np.bincount(y_train).argmax()
            
            # Create our custom DummyClassifier instance (not sklearn's)
            class DummyClassifier:
                def __init__(self, constant_class):
                    self.constant_class = constant_class
                    # Add dummy coefficients for feature importance
                    self.coef_ = np.zeros((1, X_train.shape[1]))
                    self.intercept_ = np.array([0.0])
                    
                def predict(self, X):
                    return np.full(X.shape[0], self.constant_class)
                    
                def predict_proba(self, X):
                    probs = np.zeros((X.shape[0], 2))
                    idx = 1 if self.constant_class == 1 else 0
                    probs[:, idx] = 1.0
                    return probs
                    
            model = DummyClassifier(most_common)
            cv_scores = np.array([np.mean(y_train == most_common)] * 5)
        
        return model, cv_scores
    
    def train_all_models(self):
        """
        Train models for all four prediction tasks.
        
        Returns:
        --------
        tuple: (list of trained models, X_train, X_test, y_train, y_test)
        """
        models = []
        cv_results = []
        
        # Train model for spread prediction
        spread_model, spread_cv = self.train_model(0)
        models.append(spread_model)
        cv_results.append(spread_cv)
        
        # Train model for money line prediction
        ml_model, ml_cv = self.train_model(1)
        models.append(ml_model)
        cv_results.append(ml_cv)
        
        # Train model for over/under prediction
        ou_model, ou_cv = self.train_model(2)
        models.append(ou_model)
        cv_results.append(ou_cv)
        
        # Train model for first to 15 points prediction
        first_to_15_model, first_to_15_cv = self.train_model(3)
        models.append(first_to_15_model)
        cv_results.append(first_to_15_cv)
        
        # Store models and CV results
        self.models = models
        self.cv_results = cv_results
        
        return (
            models, 
            self.processed_data['X_train'], 
            self.processed_data['X_test'], 
            self.processed_data['y_train'], 
            self.processed_data['y_test']
        )
    
    def get_feature_importance(self, model_idx):
        """
        Get feature importance scores for the specified model.
        
        Parameters:
        -----------
        model_idx : int
            0 for Spread, 1 for Money Line, 2 for Over/Under, 3 for First to 15 Points
        
        Returns:
        --------
        dict: Dictionary mapping feature names to importance scores
        """
        if self.models is None:
            raise ValueError("Models have not been trained yet")
        
        model = self.models[model_idx]
        feature_names = self.processed_data['feature_names']
        
        # Get coefficients
        coeffs = model.coef_[0]
        
        # Create a dictionary mapping features to importance scores
        importance_dict = {}
        for feature, coeff in zip(feature_names, coeffs):
            importance_dict[feature] = abs(coeff)
        
        return importance_dict
