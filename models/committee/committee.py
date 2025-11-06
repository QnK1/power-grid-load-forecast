import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class CommitteeSystem(tf.keras.Model):
    """
    A Keras model that computes its output as the average
    of predictions from a list of sub-models.
    
    This class is serializable and can be saved to .keras format.
    """
    def __init__(self, models, **kwargs):
        """
        Initializes the committee.
        
        Args:
            models (list): A list of tf.keras.Model instances
                            that form the committee (not pre-trained).
        """
        super().__init__(**kwargs)
        
        self.models = models
        self.num_models = float(len(models))
    
    
    def call(self, inputs):
        """
        The forward pass.
        """
        predictions = [model(inputs) for model in self.models]
        summed_predictions = tf.add_n(predictions)
        output = summed_predictions / self.num_models
        
        return output

    
    def get_config(self):
        """Returns the serializable configuration of the model."""
        config = super().get_config()
        
        serialized_models = [keras.saving.serialize_keras_object(m) for m in self.models]
        
        config.update({
            "models": serialized_models
        })
        return config


    @classmethod
    def from_config(cls, config):
        """Creates a model from its config."""
        deserialized_models = [keras.saving.deserialize_keras_object(m_config) 
                            for m_config in config["models"]]
        
        config["models"] = deserialized_models
        
        return cls(**config)
