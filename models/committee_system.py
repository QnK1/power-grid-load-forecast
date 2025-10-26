from utils.load_data import load_training_data, DataLoadingParams

# models/committee_system.py

import numpy as np

class CommitteeSystem:
    """
    Manages a committee of neural network models for ensemble forecasting.
    
    This class holds multiple 'expert' models. It trains them and then averages 
    their predictions to produce a final, combined forecast.
    """
    
    def __init__(self, models):
        """
        Initializes the CommitteeSystem.
        
        Args:
            models (list): A list of instantiated model objects 
                           (e.g., [MLP(), LSTM(), ...]).
        """
        if not models:
            raise ValueError("The 'models' list cannot be empty.")
            
        self.models = models
        self.name = "Committee Neural System"
        print(f"Initialized {self.name} with {len(self.models)} expert models.")

    def train(self, X_train, y_train):
        """
        Trains each model in the committee.
        
        Args:
            X_train: The training input data (features).
            y_train: The training output data (target).
        """
        print(f"--- Training the {self.name} ---")
        for i, model in enumerate(self.models):
            print(f"Training expert model {i+1}/{len(self.models)}: {model.name}...")
            # Each model is expected to have its own 'train' method.
            # We will need to define this later for each expert model.
            # For now, we'll assume it exists.
            model.train(X_train, y_train)
        print("--- All expert models have been trained. ---")

    def predict(self, X):
        """
        Makes a prediction by averaging the predictions of all models.
        
        Args:
            X: The input data for which to make predictions.
            
        Returns:
            An array of final, averaged predictions.
        """
        # Collect predictions from each model in the committee
        predictions = []
        for model in self.models:
            # Each model is expected to have its own 'predict' method.
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average the predictions
        # np.mean with axis=0 calculates the mean across the different models for each sample
        final_predictions = np.mean(predictions, axis=0)
        
        return final_predictions