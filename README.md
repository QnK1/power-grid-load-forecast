# Power grid load forecasting

## Introduction
In this project we aim to replicate the results of the paper [Bak, M., & Bielecki, A. (2007). Neural systems for short-term forecasting of electric power load. In B. Ribeiro, B. Beliczynski, A. Dz ielinski, & M. Iwanowski (Eds.), ADAPTIVE AND NATURAL COMPUTING ALGORITHMS, PT 2 (Vol. 4432, p. 133). Springer Nature.](https://link-1springer-1com-1nyztljwx006c.wbg2.bg.agh.edu.pl/book/10.1007/978-3-540-71629-7), which explores neural network–based methods for short-term power grid load prediction. In addition to reproducing the published results, we attempt to develop and evaluate a custom forecasting approach to compare its performance against the original models.

## Project structure
- `analysis` - module containing classes and methods used for data and model performance visualization, as well as the images of generated plots
- `data` - data used for model training and evaluation
- `doc` - project documentation
- `eval` - module used for models' evaluation and keeping their MAPE over horizon and average MAPE in .txt files
- `models` - module used for models' training and for storing trained models in .keras files
- `utils` - module used for data preparation and cleanup utilities

## Running the project
- Install Python 3.12.4.
- Run `pip install -r requirements.txt` to install all the necessary libraries.
- Currently our project does not have any dedicated interface for training, evaluation or plot creation. Methods responsible for those are called from temporary .py files like `trainer.py` or `evaluer.py`. We are planning on adding an actual UI to this project in the future

## Procject's conclusion
Here are the final results of average MAPE score for our models as well as the conclusions we've reached. For a more detailed version check out the `documentation`.
### Average MAPE
### Conclusions
- All models achieved single-digit MAPE values, which is the same order of magnitude as those that were reached by systems described in the paper we were using as our base.
- Models achieve better results for the first few data points in the prediction horizon, which is to be expected due to them having more "fresh" data to work with.
- Embeded systems of MLPs based on those used in the original paper (Module System, Committee System, Rule-Aided System) did deliver significantly better results compared to a single Multi-Layered Perceptron.
- LSTM's ability to preserve historical data turned out to be of little importance, as it was outperformed by less complex models.
- More advanced systems like Seq2Seq or CNN-LSTM Hybrid did not yield as satisfying results as expected (they were better than LSTMs, but still outperformed by pure CNNs). Probable cause for lack of improvement is the simple and cyclic nature of our data, which makes even simple models able to yield accurate predictions.
