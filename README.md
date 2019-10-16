# Semi-Supervised Variational Autoencoder

This repository contains code related to the paper [Semi-Supervised Variational Autoencoder for Survival Prediction](https://arxiv.org/abs/1910.04488)

We have implemented a [semi-supervised variational autoencoder](https://arxiv.org/abs/1406.5298) for modeling of 3D brain images and classification.

The code is designed for the [BraTS](http://braintumorsegmentation.org/) dataset and can be used easily after you obtain the BraTS training data.
To train a model simply run the data preparation script and then the main script.

```
python create_dataset.py --data_dir path/to/bratsdata --output_dir data/brats_19/
python main.py --data_dir_train data/brats_19/Train --data_dir_val data/brats_19/Validation
```

The code was tested with the versions provided in requirements.txt. To make sure you have them run

```
pip install -r requirements.txt
```

