import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
class Seq2SeqRepeatVectorModel(Model):
    """A sequence-to-sequence encoder-decoder model that uses LSTM layers.
    It uses the same context vector as the input for every 'step' of the recurrent decoder,
    without initializing its initial state, as opposed to classical NLP seq2seq models,
    which use the context vector as the initial hidden + cell state of the decode, running an autoregressive loop.
    """
    def __init__(self, prediction_length: int, encoder_layers: list[int], decoder_layers: list[int], bidirectional: bool, dropout, **kwargs):
        """Initializes a Seq2Seq 'RepeatVector' model.

        Args:
            prediction_length (int): Length of desired output sequences.
            encoder_layers (list[int]): List of integers. e.g., [128, 64] 
                        Creates two encoder layers: 128 units -> 64 units.
            decoder_layers (list[int]): List of integers. e.g., [64, 128]
                        Creates two decoder layers: 64 units -> 128 units.
            bidirectional (bool): Whether the model should use bidirectional LSTM layers.
                                    Each of the directions contains half of the layer's LSTM cells.
            dropout (float): Dropout percentage to apply in LSTM layers.
        """
        super(Seq2SeqRepeatVectorModel, self).__init__(**kwargs)
        
        self.prediction_length = prediction_length
        self.bidirectional = bidirectional
        self.decoder_layers = decoder_layers
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        
        self.encoder_layers_list = []
        for i, units in enumerate(encoder_layers):
            is_last_layer = (i == len(encoder_layers) - 1)
            
            if self.bidirectional:
                self.encoder_layers_list.append(
                    Bidirectional(LSTM(units // 2, return_sequences=not is_last_layer, dropout=dropout, recurrent_dropout=dropout), merge_mode='concat')
                )
            else:
                self.encoder_layers_list.append(
                    LSTM(units, return_sequences=not is_last_layer, dropout=dropout, recurrent_dropout=dropout)
                )
        
        self.encoder = Sequential(self.encoder_layers_list)

        self.repeater = RepeatVector(prediction_length)
        
        self.decoder_layers_list = []
        for i, units in enumerate(decoder_layers):
            if self.bidirectional:
                self.decoder_layers_list.append(
                    Bidirectional(LSTM(units // 2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout), merge_mode='concat')
                )
            else:
                self.decoder_layers_list.append(
                    LSTM(units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
                )
        
        self.decoder = Sequential(self.decoder_layers_list)
            
        self.output_head = TimeDistributed(Dense(1))


    def call(self, inputs):
        context = self.encoder(inputs)

        repeated_context = self.repeater(context)

        decoder_output = self.decoder(repeated_context)

        return self.output_head(decoder_output)
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "prediction_length": self.prediction_length,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        })
        return config


    def build(self, input_shape):
        self.encoder.build(input_shape)
        
        encoder_output_shape = self.encoder.compute_output_shape(input_shape)
        
        self.repeater.build(encoder_output_shape)
        repeater_output_shape = self.repeater.compute_output_shape(encoder_output_shape)
        
        self.decoder.build(repeater_output_shape)
        decoder_output_shape = self.decoder.compute_output_shape(repeater_output_shape)
        
        self.output_head.build(decoder_output_shape)
        
        super().build(input_shape)
