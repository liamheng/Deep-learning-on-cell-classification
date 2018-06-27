# Deep-learning-models
These files is to perform deep learning models on cell classification. The packages of Keras, Tensorflow, os, xgboost and sklearn are necessary to use the code.
Please uncompress the file of ExampleDataset.rar, and put the output ExampleDataset.npz in the same folder of the .py files.

## Example data
ExampleDataset.rar is the compressed file of ExampleDataset.npz, which contains 30 cell sequences for binary classification. For a cell video, it is represented by the contour sequence and then is zoomed to size of 224x224. Augmentation is performed to generate 216 data subjects from one sequence. Therefore the shape of the example dataset is 30x216x224x224.

<img alt="scratch" src="images/Sequence.png" width='450'>  
<sub><b>Figure 1: </b> Cell sequence to contour spectrum. </sub> 

## Deep learning frameworks
1. ScratchModel.py trains a CNN model from scratch for cell classification.

<img alt="scratch" src="images/Structure.png" width='600'>  

<sub><b>Figure 2: </b> Scratch model structure. </sub> 

2. VggFeatures.py extracts deep features of cell sequence with pre-trained Vgg models. And then SVM and XGboost are performed.

<img alt="scratch" src="images/Feature.png" width='500'>  

<sub><b>Figure 2: </b> CNN Feature structure. </sub> 

3. VggModel.py fine-tunes pre-trained Vgg models to classify cells.

<img alt="scratch" src="images/VGG.png" width='500'>  

<sub><b>Figure 3: </b> Fine-tuning structure. </sub> 
