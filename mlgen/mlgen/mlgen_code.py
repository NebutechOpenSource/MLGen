import re 
import sys
import yaml
import json
import os
import pkg_resources

class code_generate:
    def __init__(self):
        self.imports = ""
        self.reqs = ""
        self.data = ""
        self.mlcode = ""
        self.checkpointcode = ""
        self.compilecode = ""
        self.savecode = ""
        self.pygen = None
        self.reqgen = None
        self.selectedColoumns = None
        self.Lable = None
        self.jsondata = json.load(open("mlm.json",'r'))
        self.dictS = yaml.load(open(self.jsondata['filename']))
        self.split = ""

        self.genKeys={
            'type': self.findType,
            'data': self.dataFill,
            'split': self.createSplit,
            'linear_regression': self.linear_regression,
            'CNN': self.findCNN,
            'ANN': self.findANN,
            'LSTM': self.findLSTM,
            'metrix':self.metrix,
            'compile':self.model_complie_fit,
            'checkpoint':self.model_checkpoint,
            'save_model': self.savemodel,
        }   


    def findType(self):
        typedict={
            'jupyter': self.createfileJupyter,
            'python': self.createfile
        }

        typedict[self.dictS['type']]()
            

    
    def createfileJupyter(self):
        print("creating jupyter files")
        self.pygen=open(f"{self.dictS['file'].split('.')[0]}.ipynb",'w')
        self.reqgen=open("requirments.txt",'w')
        pythonfile = {
            'pythonfile': f"{self.dictS['file'].split('.')[0]}.ipynb"
        }

        with open("mlm.json", "r") as read_file:
            data = json.load(read_file)
            pythonfile['filename'] = data['filename']
          
        with open(f'mlm.json', 'w') as json_file:
            json.dump(pythonfile, json_file,sort_keys=True, indent=4)


    def createfile(self):
        print("creating python")
        self.pygen=open(f"{self.dictS['file'].split('.')[0]}.py",'w')
        self.reqgen=open("requirments.txt",'w')
        pythonfile = {
            'pythonfile': f"{self.dictS['file'].split('.')[0]}.py"
        }


        with open("mlm.json", "r") as read_file:
            data = json.load(read_file)
            pythonfile['filename'] = data['filename']
            

        with open(f'mlm.json', 'w') as json_file:
            json.dump(pythonfile, json_file,sort_keys=True, indent=4)
        
        self.imports += f"#!/usr/local/bin/python{self.dictS['version']}\n\n"

    def dataFill(self):
        print("filling import data")
        self.imports +="import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
        self.reqs += "pandans\nmatplotlib\n"
        try:
            self.imports += "import request\nimport io\n"
            setlocation = re.search("(?P<url>https?://[^\s]+)", self.dictS['data']).group("url")
            self.data += f"s=requests.get({setlocation}).content\nndataFile=pd.read_csv(io.StringIO(s.decode('utf-8')))"

        except:
            self.data += f"\ndataFile = \'{self.dictS['data']}\'\n"
        

    def createSplit(self):
        print("creating data split")
        self.imports +="from sklearn.model_selection import train_test_split\n"
        self.reqs += "sklearn\n"
        self.split += f"\nx_train,x_test,y_train,y_test=train_test_split(features,lable,test_size={int(self.dictS['split'])/100})\n"

    def linear_regression(self):
        self.coloumns_extract()
        path = 'template/linreg.txt'
        filepath = pkg_resources.resource_filename(__name__, path)

        with open(filepath, 'rb') as fi:
            for i in fi.readlines():
                self.data += i
        self.metrix()
        
        self.imports += f"from sklearn import linear_model\n"
    
    def coloumns_extract(self):
        print("extracting coloumns")
        col = [] 
        for i in self.dictS['coloumns_feature']:
            col.append(i)
        self.data +=f"dataFrame = pd.read_csv(dataFile)\nfeatures = dataFrame[{col}]\n"
        col =[]
        for i in self.dictS['coloumns_lable']:
            col.append(i)
        self.data +=f"lable = dataFrame[{col}]\n\n"
        self.data_preprocessing()
        self.data += self.split



    def data_preprocessing(self):  
        preprodict={
            'remove': self.datapro_remove,
            'mean': self.datapro_mean,
            'median': self.datapro_median,
            'mode':self.datapro_mode,
            'nill_data_categorical':self.datapro_nill_data_categorical
        } 
        self.datapro_changeType()
        preprodict[self.dictS['nill_data']]()
        

    def metrix(self):
        for i in self.dictS['linear_regression']['metrix']:
            self.data += f"print(\'{i} %.2f:\' % {i}(y_test, y_pred))\n"
            self.imports += f"from sklearn.metrics import {i}\n"

    def datapro_changeType(self):
        for i in self.dictS['coloumns_feature']:
            if self.dictS['coloumns_feature'][i]['type'] != 'categorical':
                self.data += f"features[\'{i}\'] = features[\'{i}\'].astype({self.dictS['coloumns_feature'][i]['type']})\n"
            
            else:
                self.data += f"features[\'{i}\'] = features[\'{i}\'].astype('categorical').cat.codes\n"
        
        for i in self.dictS['coloumns_lable']:
            if self.dictS['coloumns_lable'][i]['type'] != 'categorical':
                self.data += f"lable[\'{i}\'] = lable[\'{i}\'].astype({self.dictS['coloumns_lable'][i]['type']})\n"
            
            else:
                self.data += f"lable[\'{i}\'] = lable[\'{i}\'].astype('categorical').cat.codes\n"
    
    
    def datapro_remove(self):
        self.data += f"dataFrame = dataFrame.dropna(how='any',axis=0)\n\n"  
            
    def datapro_mean(self):
        for i in self.dictS['coloumns_feature']:
            if self.dictS['coloumns_feature'][i]['type'] != 'categorical':
                self.data += f"features[\'{i}\'] = features[\'{i}\'].replace(np.NaN,features[\'{i}\'].mean())\n"
            else:
                self.datapro_nill_data_categorical()
        for i in self.dictS['coloumns_lable']:
            if self.dictS['coloumns_lable'][i]['type'] != 'categorical':
                self.data += f"lable[\'{i}\'] = lable[\'{i}\'].replace(np.NaN,lable[\'{i}\'].mean())\n"
            else:
                self.datapro_nill_data_categorical()

    def datapro_median(self):
        for i in self.dictS['coloumns_feature']:
            if self.dictS['coloumns_feature'][i]['type'] != 'categorical':
                self.data += f"features[\'{i}\'] = features[\'{i}\'].replace(np.NaN,features[\'{i}\'].median())\n"
            else:
                self.datapro_nill_data_categorical()
        for i in self.dictS['coloumns_lable']:
            if self.dictS['coloumns_lable'][i]['type'] != 'categorical':
                self.data += f"lable[\'{i}\'] = lable[\'{i}\'].replace(np.NaN,lable[\'{i}\'].median())\n"
            else:
                self.datapro_nill_data_categorical()

    def datapro_mode(self):
        for i in self.dictS['coloumns_feature']:
            if self.dictS['coloumns_feature'][i]['type'] != 'categorical':
                self.data += f"features[\'{i}\'] = features[\'{i}\'].replace(np.NaN,features[\'{i}\'].mode())\n"
            else:
                self.datapro_nill_data_categorical()

        for i in self.dictS['coloumns_lable']:
            if self.dictS['coloumns_lable'][i]['type'] != 'categorical':
                self.data += f"lable[\'{i}\'] = lable[\'{i}\'].replace(np.NaN,lable[\'{i}\'].mode())\n"
            else:
                self.datapro_nill_data_categorical()

    def datapro_nill_data_categorical(self):
        for i in self.dictS['coloumns_feature']:
            if self.dictS['coloumns_feature'][i]['type'] == 'categorical':
                self.data += f"features[\'{i}\'] = features.replace(np.NaN,features['{i}'].value_counts().idx{self.dictS['nill_data_categorical']}())\n"
        
        for i in self.dictS['coloumns_lable']:
            if self.dictS['coloumns_lable'][i]['type'] == 'categorical':
                self.data += f"lable[\'{i}\'] = lable.replace(np.NaN,lable['{i}'].value_counts().idx{self.dictS['nill_data_categorical']}())\n"
        

    def reshape(self):
        self.data += f"x_train = x_train.reshape((x_train.len(),{self.dictS['reshape']}))\nx_test = x_test.reshape(x_train.reshape((x_train.len(),{self.dictS['reshape']}))\n"

    
    def findCNN(self):
        CNN_dict={
            'tensorflow 2.0':self.genCNN_tensorflow2,
            'keras': self.genCNN_keras
        }

        CNN_dict[self.dictS['backend']]()

    def findANN(self):
        ANN_dict={
            'tensorflow 2.0':self.genANN_tensorflow2,
            'keras': self.genANN_keras
        }


        ANN_dict[self.dictS['backend']]()

    def findLSTM(self):
        LSTM_dict = {
            'tensorflow 2.0':self.genLSTM_tensorflow2,
            'keras': self.genLSTM_keras
        }
        LSTM_dict[self.dictS['backend']]()


    def genCNN_tensorflow2(self):
        print("generating tensorflow CNN files")
        self.coloumns_extract()
        self.reshape()
        if self.dictS['gpu'] == 'true':
            self.reqs += "tensorflow-gpu==2.0.0-beta1\n"
        else:
            self.reqs += "tensorflow==2.0.0-beta1\n"
        no = 1
        self.mlcode += "\nmodel = models.Sequential()\n\n"
        self.imports += "import tensorflow as tf\nfrom tensorflow.keras import layers, models\n" 
        
        for i in self.dictS['CNN']:
                
            if no == 1:
                input_shape = (0,0,0)
                if 'input_shape' in self.dictS['CNN'][i]:
                    input_shape = self.dictS['CNN'][i]['input_shape']

                if 'maxpool' not in self.dictS['CNN'][i]:
                    maxpool = (2,2)
                    no=0
                else:
                    maxpool = self.dictS['CNN'][i]['maxpool']
                
                self.mlcode += f"model.add(layers.Conv2D({self.dictS['CNN'][i]['number_neurons']},{self.dictS['CNN'][i]['kernal']}, activation='relu', input_shape={input_shape}))\n"
                self.mlcode += f"model.add(layers.MaxPooling2D({maxpool}))\n"
            else:
                if 'maxpool' not in self.dictS['CNN'][i]:
                    maxpool = (3,3)
                else:
                    maxpool = self.dictS['CNN'][i]['maxpool']
                
                self.mlcode += f"model.add(layers.Conv2D({self.dictS['CNN'][i]['number_neurons']},{self.dictS['CNN'][i]['kernal']}, activation='relu'))\n"
                self.mlcode += f"model.add(layers.MaxPooling2D({maxpool}))\n"
        
        self.data += "\nmodel.summary()\n\n"

        for i in self.dictS['Dense']:
            self.mlcode += f"model.add(layers.Dense({self.dictS['Dense'][i]['number_neurons']}, activation={self.dictS['Dense'][i]['activation']}))\n"
    
        self.mlcode += "\nmodel.summary()\n\n"


    def genCNN_keras(self):
        print("generating keras CNN files")
        self.coloumns_extract()
        
        self.reshape()
        if self.dictS['gpu'] == 'true':
            self.reqs += "tensorflow-gpu\nkeras\n"
            
        else:
            self.reqs += "keras\n"
        no = 1
        self.mlcode += "\nmodel = models.Sequential()\n\n"
        self.imports += "from keras import models\nfrom keras.layers import Dense, Conv2D, Flatten\n" 
        
        for i in self.dictS['CNN']:

            if no == 1:
                input_shape = (0,0,0)
                if 'input_shape' in self.dictS['CNN'][i]:
                    input_shape = self.dictS['CNN'][i]['input_shape']
                if 'maxpool' not in self.dictS['CNN'][i]:
                    maxpool = (2,2)
                    no=0
                else:
                    maxpool = self.dictS['CNN'][i]['maxpool']
                
                self.data += f"model.add(layers.Conv2D({self.dictS['CNN'][i]['number_neurons']},{self.dictS['CNN'][i]['kernal']}, activation='relu', input_shape={input_shape}))\n"
                self.data += f"model.add(layers.MaxPooling2D({maxpool}))\n"
            else:
                if 'maxpool' not in self.dictS['CNN'][i]:
                    maxpool = (3,3)
                else:
                    maxpool = self.dictS['CNN'][i]['maxpool']
                
                self.data += f"model.add(layers.Conv2D({self.dictS['CNN'][i]['number_neurons']},{self.dictS['CNN'][i]['kernal']}, activation='relu'))\n"
                self.data += f"model.add(layers.MaxPooling2D({maxpool}))\nmodel.add(layers.Flatten())\n"
        
        self.mlcode += "\nmodel.summary()\n\n"
            
        for i in self.dictS['Dense']:
            self.mlcode += f"model.add(layers.Dense({self.dictS['Dense'][i]['number_neurons']}, activation=\'{self.dictS['Dense'][i]['activation']}\'))\n"
        
        self.data += "\nmodel.summary()\n\n"

    def genANN_keras(self):
        print("generating keras ANN files")
        self.coloumns_extract()
        self.data_preprocessing()
        if self.dictS['gpu'] == 'true':
            self.reqs += "tensorflow-gpu\nkeras"
        else:
            self.reqs += "tensorflow\nkeras"
        no = 1
        self.mlcode += "\nmodel = models.Sequential()\n\n"
        self.imports += "from keras import models\nfrom keras.layers import Dense, Dropout\n" 
        
        for i in self.dictS['ANN']:
            input_dim = 10
            self.dropout = 0
            self.noise_shape = None
            self.seed = None

            if no == 1:
                self.dropoutGen('ANN',i)

                self.mlcode += f"model.add(Dense({self.dictS['ANN'][i]['number_neurons']}, input_dim={input_dim}, activation=\'{self.dictS['ANN'][i]['activation']}\'))\n"
                self.mlcode += f"model.add(Dropout({self.dictS['ANN'][i]['dropout']['dropout']}, noise_shape={self.noise_shape}, seed={self.seed}))\n"
                no = 0

            else:
                self.dropoutGen('ANN',i)
                self.mlcode += f"model.add(Dense({self.dictS['ANN'][i]['number_neurons']}, activation=\'{self.dictS['ANN'][i]['activation']}\'))\n"
                self.mlcode += f"model.add(Dropout({self.dictS['ANN'][i]['dropout']['dropout']}, noise_shape={self.noise_shape}, seed={self.seed}))\n"
    
    def genANN_tensorflow2(self):  
        print("generating tensorflow ANN files")
        self.coloumns_extract()
        self.data_preprocessing()
        if self.dictS['gpu'] == 'true':
            self.reqs += "tensorflow-gpu==2.0.0-beta1\n"
        else:
            self.reqs += "tensorflow==2.0.0-beta1\n"
        no = 1
        self.mlcode += "\nmodel = models.Sequential()\n\n"
        self.imports += "from keras import models\nfrom keras.layers import Dense, Dropout\n" 
        
        for i in self.dictS['ANN']:
            input_dim = 10
            self.dropout = 0
            self.noise_shape = None
            self.seed = None

            if no == 1:
                self.dropoutGen('ANN',i)

                self.mlcode += f"model.add(Dense({self.dictS['ANN'][i]['number_neurons']}, input_dim={input_dim}, activation=\'{self.dictS['ANN'][i]['activation']}\'))\n"
                self.mlcode += f"model.add(Dropout({self.dictS['ANN'][i]['dropout']['dropout']}, noise_shape={self.noise_shape}, seed={self.seed}))\n"
                no = 0

            else:
                self.dropoutGen('ANN',i)
                self.mlcode += f"model.add(Dense({self.dictS['ANN'][i]['number_neurons']}, activation=\'{self.dictS['ANN'][i]['activation']}\'))\n"
                self.mlcode += f"model.add(Dropout({self.dictS['ANN'][i]['dropout']['dropout']}, noise_shape={self.noise_shape}, seed={self.seed}))\n"

    def genLSTM_tensorflow2(self):
        print("generating tensorflow LSTM files")
        if self.dictS['gpu'] == 'true':
            self.reqs += "tensorflow-gpu==2.0.0-beta1\n"
        else:
            self.reqs += "tensorflow==2.0.0-beta1\n"

    
    def genLSTM_keras(self):
        self.coloumns_extract()
        self.data_preprocessing()
        if self.dictS['gpu'] == 'true':
            self.reqs += "tensorflow-gpu\nkeras\n"
        else:
            self.reqs += "keras\n"
        no = 1
        self.imports += "from keras.models import Sequential\nfrom keras.layers import Dense, Dropout,LSTM,Activation"
        self.mlcode += "\n\nmodel = Sequential()\n"
        for i in self.dictS['LSTM']:
            return_sequences = True if 'return_sequences' in self.dictS['LSTM'][i] else False

            if no == 1:
                input_shape = (0,0,0)
                if 'input_shape' in self.dictS['LSTM'][i]:
                    input_shape = self.dictS['LSTM'][i]['input_shape']

                self.mlcode += f"model.add(LSTM({self.dictS['LSTM'][i]['number_neurons']},input_shape={input_shape},return_sequences={return_sequences}))\n"
                no = 0
                self.dropoutGen('LSTM',i)
                self.mlcode += f"model.add(Dropout({self.dictS['LSTM'][i]['dropout']['dropout']}, noise_shape={self.noise_shape}, seed={self.seed}))\n"
               

            else:
                self.mlcode += f"model.add(LSTM({self.dictS['LSTM'][i]['number_neurons']}, return_sequences=\'{return_sequences}\'))\n"
                self.dropoutGen('LSTM',i)
                self.mlcode += f"model.add(Dropout({self.dictS['LSTM'][i]['dropout']['dropout']}, noise_shape={self.noise_shape}, seed={self.seed}))\n"
        
        for i in self.dictS['Dense']:
            self.mlcode += f"model.add(layers.Dense({self.dictS['Dense'][i]['number_neurons']}, activation=\'{self.dictS['Dense'][i]['activation']}\'))\n"            


    def dropoutGen(self,neunet,i):
        self.dropout = self.dictS[neunet][i]['dropout']['dropout']
        if 'noise_shape' in self.dictS[neunet][i]['dropout']:
            self.noise_shape = self.dictS[neunet][i]['dropout']['noise_shape']
        if 'seed' in self.dictS[neunet][i]['dropout']:
            self.seed = self.dictS[neunet][i]['dropout']['seed']


    def model_complie_fit(self):
        batch_size = 10
        verbose = 0
        optarr = []

        for i in self.dictS['compile']['metrics']:
            optarr.append(i)

        self.compilecode += f"\nmodel.compile(optimizer=\'{self.dictS['compile']['optimizer']}\',loss=\'{self.dictS['compile']['loss']}\',metrics={optarr})\n"
        
        if 'verbose' in self.dictS['compile']:
            verbose = self.dictS['compile']['batch_size']
        if 'batch_size' in self.dictS['compile']:
            batch_size = self.dictS['compile']['batch_size']
        
        self.compilecode += f"\nmodel.fit(x_train, y_train, epochs={self.dictS['compile']['epochs']}, batch_size={batch_size}, verbose={verbose})"

    def model_checkpoint(self):
        self.imports += "from keras.callbacks import ModelCheckpoint\n"
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

        verbose = 0 if 'verbose' not in self.dictS['compile'] else self.dictS['compile']['verbose']

        monitor = 'loss' if 'monitor' not in self.dictS['compile'] else self.dictS['compile']['monitor']
        save_best_only = True if 'save_best_only' not in self.dictS['compile'] else self.dictS['compile']['save_best_only']
        mode = 'min' if 'save_best_only' not in self.dictS['compile'] else self.dictS['compile']['mode']
        self.checkpointcode += f"\nfilepath = \'{filepath}\'"
        self.checkpointcode += f"\ncheckpoint = ModelCheckpoint(filepath,monitor=\'{monitor}\',verbose={verbose},save_best_only={save_best_only},mode=\'{mode}\')\n"

    def savemodel(self):
        print("saving model")
        self.reqs += "h5py\n"
        self.imports += "from keras.models import model_from_json, load_model\n"
        self.savecode += "\n#saving the model\n"
        
        if 'weights' in self.dictS['save_model']['save']:
            self.savecode += f"model.save_weights(\'{self.dictS['save_model']['file']}.h5\')"
        if 'model' in self.dictS['save_model']['save']:
            self.savecode += f"model.save(\'{self.dictS['save_model']['file']}.h5\')"

    def generatefiles(self):
        for i in self.dictS:
            try:
                self.genKeys[i]()
            except:
                pass

        if self.dictS['type'] == 'python':
            self.pygen.writelines([self.imports,self.data,self.mlcode,self.compilecode,self.checkpointcode,self.savecode])
            self.reqgen.writelines([self.reqs])


        if self.dictS['type'] == 'jupyter':
            path = '/template/python3_jup.ipynb'
            filepath = pkg_resources.resource_filename(__name__, path)
            with open(filepath, 'rb') as f:
                jf = json.load(f)

            for i in jf['cells']:
                var = i['metadata']['type']
                if var == 'importsinstall':
                    req = self.reqs.split("\n")
                    req = list(filter(None, req))
                    for e in req:
                        i['source'].append(f"!pip install {e}\n")
                if var == 'imports':
                    i['source'].append(self.imports)
                if var == 'data':
                    i['source'].append(self.data)
                if var == 'mlcode':
                    i['source'].append(self.mlcode)
                if var == 'compilecode':
                    i['source'].append(self.compilecode)
                if var == 'checkpointcode':
                    i['source'].append(self.checkpointcode)
                if var == 'savecode':
                    i['source'].append(self.savecode)

            with open(f"{self.dictS['file'].split('.')[0]}.ipynb", 'w') as outfile:
                json.dump(jf, outfile,sort_keys=True, indent=4)

            self.reqgen.writelines([self.reqs])
        self.pygen.close()
        self.reqgen.close()
                        