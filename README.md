# Deep-learning-models
Deep learning models for cell classification

ExampleDataset.rar is the compressed file of ExampleDataset.npz, which contains 30 cell sequences for binary classification. For a cell video, it is represented by the contour sequence and then is zoomed to size of 224x224. Augmentation is performed to generate 216 data subjects from one sequence. Therefore the shape of example dataset is 30x216x224x224.

ScratchModel.py trains a CNN model from scratch for cell classification.

VggFeatures.py extracts deep features of cell sequence with pre-trained Vgg models. And then SVM and XGboost are performed.

VggModel.py fine-tunes pre-trained Vgg models to classify cells.
