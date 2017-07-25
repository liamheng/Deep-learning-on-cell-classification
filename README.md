# Deep-learning-models
Deep learning models for cell classification

ExampleDataset.rar is the compressed file of ExampleDataset.npz, which contains 30 cell sequences for binary classification.

ScratchModel.py trains a CNN model from scratch for cell classification.

VggFeatures.py extracts deep features of cell sequence with pre-trained Vgg models. And then SVM and XGboost are performed.

VggModel.py fine-tunes pre-trained Vgg models to classify cells.
