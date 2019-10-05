# MLGen

[![Join the chat at https://gitter.im/MLGen/community](https://badges.gitter.im/MLGen/community.svg)](https://gitter.im/MLGen/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


# MLGen: Machine learning code generator

[![Join the chat at https://gitter.im/MLGen/community](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/MLGen/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.png?v=103)](https://opensource.org/licenses/mit-license.php)  
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


MlGen is a tool which helps you to generate machine learning code with ease.

MLGen uses a ".mlm" file format which is a file with YML like syntax.

This tool as of now supports keras and tensorflow2.0(not fully supported) and more yet to come....


## Getting Started
### Installing
  
`pip install mlgen`

### Upgrading the package
` pip install --upgrade mlgen`
  

## CLI commands

__To initialize a file in your workspace__

`mlgen -i | --init <file name>`

__To generate a specific template (optional)__

`mlgen -g | --gen <neural network type> --backend | -be <lib to use> -t python | jupyter`

__To generate the ml python file__

`mlgen -r . `

  

## MLM file syntax

  

**file**: name of the python file to be created

  

**version**: version of python being used

  

**backend**: which machine learning platform if to be used

  

**gpu**: (bool) is gpu being used or not

  

**data**: location of the dataset can be a URL/ folder location on machine

  

**split**:(int) slipt in training and testing data. automatically converted to a decimal

  

**coloumns_feature**: list of coloumns being used for the prediction

**nill_data**: basic null data handling in non categorical datatypes. Available techiniques remove, mean, mode, median

**nill_data_categorical**: basic null data handling for categorical datatypes. Available techiniques remove, max, min

**NeuralNetwork_type**: the type of neural network being used such as ANN, CNN, LSTM

#### layers section:
**number_neurons**: (int) number of neurons

**input_dim**: input dimensions of the first layer input

**activation**: activation function being used

**dropout**: (int) dropout percentage

**noise_shape**: (int) noise shape (optional)

**seed**: (int) seed value (optional)

#### compile section:
**epochs**: (int) number of epoch

**batch_size**: (int) batch size

**verbose**: (int) verbose value 0,1,2

**optimizer**: optimizer being used

**loss**: loss type

**metrics**: (array) represents a metric such as accuracy

#### checkpoint section: (optional)

**monitor**: metrix type

**verbose**: (int) batch size

**save_best_only**: (bool)

**mode**: mode such as min max


#### save_model section: (optional)

**file**: file name to save model in

**save**: save type. Available options weights and model

## Eample mlm file
`mlgen -i anniris`  
`mlgen -g ann -be keras -t jupyter`

*make nessesary changes to the mlm file*
<pre>
file: anniris
version: 3.7
type: python
backend: keras
gpu: true
data: data.csv

split: 70

coloumns_feature:
  sepal_length:
    type: float
  sepal_width:
    type: float
  petal_length:
    type: float
  petal_width:
    type: float

coloumns_lable:
  species:
    type: categorical

nill_data: mean
nill_data_categorical: max

ANN:
  layer1:
    number_neurons: 12
    input_dim: 8
    activation: relu
    dropout:
      dropout: 70
      noise_shape: 12
      seed: 12
  layer2:
    number_neurons: 8
    activation: relu
    dropout:
      dropout: 70
      noise_shape: 12
      seed: 12
  layer3:
    number_neurons: 1
    activation: sigmoid

compile:
  epochs: 5
  batch_size: 10
  verbose: 0
  optimizer: adam
  loss: binary_crossentropy
  metrics:
  - accuracy

checkpoint:
  monitor: loss
  verbose: 1
  save_best_only: True,
  mode: min

save_model:
  file: model
  save: model
</pre>

to generate jupyter notebook 
`mlgen -r .`



### Development by
Company: [Nebutech](https://github.com/NebutechOpenSource/)

### Contact / Social Media

*Get the latest News about Web Development, Open Source, Tooling, Server & Security*

[![Gitter](https://github.frapsoft.com/social/gitter.png)](https://gitter.im/MLGen/community)[![Github](https://github.frapsoft.com/social/github.png)](https://github.com/NebutechOpenSource/)

