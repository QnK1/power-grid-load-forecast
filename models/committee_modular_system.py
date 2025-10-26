# models/committee_modular_system.py

import numpy as np

class CommitteeSystem:
    """
    Manages a committee of neural network models for ensemble forecasting.
    This class holds multiple 'expert' models, trains them, and averages 
    their predictions to produce a final, combined forecast.
    """
    def __init__(self, models):
        """
        Initializes the CommitteeSystem.
        Args:
            models (list): A list of instantiated model objects that have
                           .train() and .predict() methods.
        """
        if not models:
            raise ValueError("The 'models' list cannot be empty.")
            
        self.models = models
        self.name = "Committee Neural System"
        print(f"Initialized {self.name} with {len(self.models)} expert models.")


    def train(self, X_train, y_train):
        """
        Trains each model in the committee on the provided data.
        Args:
            X_train: The training input data (features).
            y_train: The training output data (target).
        """
        print(f"--- Training the {self.name} ---")
        for i, model in enumerate(self.models):
            # Check if model object has a name attribute before trying to print it
            model_name = getattr(model, 'name', f'model_{i+1}') 
            print(f"Training expert model {i+1}/{len(self.models)}: {model_name}...")
            model.train(X_train, y_train)
        print("--- All expert models have been trained. ---")


    def predict(self, X):
        """
        Generates a prediction by averaging the outputs of all models.
        Args:
            X: The input data for which to make predictions.
        Returns:
            np.ndarray: An array containing the final, averaged predictions.
        """
        # Collect predictions from each model in the committee
        all_predictions = [model.predict(X) for model in self.models]
        
        # Average the predictions across the models for each sample
        final_predictions = np.mean(all_predictions, axis=0)
        
        return final_predictions