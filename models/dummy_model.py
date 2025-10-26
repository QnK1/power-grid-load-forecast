# models/dummy_model.py

import numpy as np
import time

class DummyModel:
    """
    A simple placeholder model for testing the CommitteeSystem.
    
    This model does not perform any real training. It simply pretends to train
    and generates random predictions. Its purpose is to have the same methods
    (train, predict) and attributes (name) as the real models will.
    """
    
    def __init__(self, name="Dummy Model"):
        """
        Initializes the DummyModel.
        
        Args:
            name (str): The name of this dummy model instance.
        """
        self.name = name
        print(f"Initialized {self.name}.")

    def train(self, X_train, y_train):
        """
        Simulates the training process.
        """
        print(f"   '{self.name}' is pretending to train...")
        # Simulate work by pausing for a second
        time.sleep(1)
        print(f"   '{self.name}' training complete.")

    def predict(self, X):
        """
        Generates random predictions.
        
        The output shape must match the target variable's shape. Since our target 
        is a single value (the power load), the shape will be (number of samples, 1).
        
        Args:
            X: The input data.
            
        Returns:
            A numpy array of random predictions.
        """
        # Generate random predictions between 40000 and 70000
        num_samples = len(X)
        predictions = 40000 + np.random.rand(num_samples, 1) * 30000
        return predictions