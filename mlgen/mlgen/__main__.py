import argparse
import sys
import yaml
from .mlgen_code import code_generate
import os
import pkg_resources
import json

def main():
    parser = argparse.ArgumentParser(description="Generate machine learning files in either python or jupyter notebook formats",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--info',
                    help='information and additional resources of cli',
                    )
    parser.add_argument('--init','-i', type=str,
                    help='initialize files',metavar='init')
    parser.add_argument('--gen','-g', type=str,
                help='generate ML model\'s mlm files')       
    parser.add_argument('--backend','-be', type=str,
                help='backend ML framework used to generate files')       
    parser.add_argument('--type','-t', type=str, default = 'python',
                help='python or jupyter file being generated')       
    parser.add_argument('--run','-r', type=str, 
                    help='run mlm file to generate python code')


    args = parser.parse_args()
    
    sys.stdout.write(str(genfiles(args)))

def createfile(name):
    print("creating files")
    open(f"{name}.mlm", 'w+').close()
    filename = {
        'filename': f'{name}.mlm'
    }
    with open(f'mlm.json', 'w') as json_file:
        json.dump(filename, json_file,sort_keys=True, indent=4)
    print("file name",filename)
    return "creating"

def readfile():
    print("reading files")
    x = json.load(open("mlm.json",'r'))
    print(x['filename'])
    return x['filename']

def genfiles(args):
    if args.info:
        print(
    """
            
                                                                                                             
                                                                                                             
            MMMMMMMM               MMMMMMMLLLLLLLLLLL                   GGGGGGGGGGGGG                                    
            M:::::::M             M:::::::L:::::::::L                GGG::::::::::::G                                    
            M::::::::M           M::::::::L:::::::::L              GG:::::::::::::::G                                    
            M:::::::::M         M:::::::::LL:::::::LL             G:::::GGGGGGGG::::G                                    
            M::::::::::M       M::::::::::M L:::::L              G:::::G       GGGGGG   eeeeeeeeeeee   nnnn  nnnnnnnn    
            M:::::::::::M     M:::::::::::M L:::::L             G:::::G               ee::::::::::::ee n:::nn::::::::nn  
            M:::::::M::::M   M::::M:::::::M L:::::L             G:::::G              e::::::eeeee:::::en::::::::::::::nn 
            M::::::M M::::M M::::M M::::::M L:::::L             G:::::G    GGGGGGGGGe::::::e     e:::::nn:::::::::::::::n
            M::::::M  M::::M::::M  M::::::M L:::::L             G:::::G    G::::::::e:::::::eeeee::::::e n:::::nnnn:::::n
            M::::::M   M:::::::M   M::::::M L:::::L             G:::::G    GGGGG::::e:::::::::::::::::e  n::::n    n::::n
            M::::::M    M:::::M    M::::::M L:::::L             G:::::G        G::::e::::::eeeeeeeeeee   n::::n    n::::n
            M::::::M     MMMMM     M::::::M L:::::L         LLLLLG:::::G       G::::e:::::::e            n::::n    n::::n
            M::::::M               M::::::LL:::::::LLLLLLLLL:::::LG:::::GGGGGGGG::::e::::::::e           n::::n    n::::n
            M::::::M               M::::::L::::::::::::::::::::::L GG:::::::::::::::Ge::::::::eeeeeeee   n::::n    n::::n
            M::::::M               M::::::L::::::::::::::::::::::L   GGG::::::GGG:::G ee:::::::::::::e   n::::n    n::::n
            MMMMMMMM               MMMMMMMLLLLLLLLLLLLLLLLLLLLLLLL      GGGGGG   GGGG   eeeeeeeeeeeeee   nnnnnn    nnnnnn
                                                                                                            

    """
        )

        print("Genrate ML files in python and jupyter with this tool\n")
        #print("Learn more: https://mlgen.com")
        print("Contribute: https://github.com/NebutechOpenSource/MLGen")
        #print("Learn more about Nebutech: https://nebutech.in\n")

        print(
            """
            /$$   /$$                                           /$$   /$$                  /$$      /$$                  
            | $$  | $$                                          | $$  | $$                 | $$     |__/                  
            | $$  | $$ /$$$$$$  /$$$$$$  /$$$$$$ /$$   /$$      | $$  | $$ /$$$$$$  /$$$$$$| $$   /$$/$$/$$$$$$$  /$$$$$$ 
            | $$$$$$$$|____  $$/$$__  $$/$$__  $| $$  | $$      | $$$$$$$$|____  $$/$$_____| $$  /$$| $| $$__  $$/$$__  $$
            | $$__  $$ /$$$$$$| $$  \ $| $$  \ $| $$  | $$      | $$__  $$ /$$$$$$| $$     | $$$$$$/| $| $$  \ $| $$  \ $$
            | $$  | $$/$$__  $| $$  | $| $$  | $| $$  | $$      | $$  | $$/$$__  $| $$     | $$_  $$| $| $$  | $| $$  | $$
            | $$  | $|  $$$$$$| $$$$$$$| $$$$$$$|  $$$$$$$      | $$  | $|  $$$$$$|  $$$$$$| $$ \  $| $| $$  | $|  $$$$$$$
            |__/  |__/\_______| $$____/| $$____/ \____  $$      |__/  |__/\_______/\_______|__/  \__|__|__/  |__/\____  $$
                              | $$     | $$      /$$  | $$                                                       /$$  \ $$
                              | $$     | $$     |  $$$$$$/                                                      |  $$$$$$/
                              |__/     |__/      \______/                                                        \______/ 
            """
        )

    if args.init is not None:
        print("args",args.init)
        createfile(args.init)
        print("init files")

    if args.run:
        print("running files")
        par = code_generate()
        par.generatefiles()
        
    
    if args.gen:

        if args.gen == 'cnn':
            if args.backend == 'tensorflow2.0':
                print("generating tf files")
                path = '/mlm_templates/cnn_tensorflow2.mlm'
                filepath = pkg_resources.resource_filename(__name__, path)
                yamlfile = yaml.load(open(filepath))
                myFile = readfile()
                yamlfile['file'] = myFile.split('.')[0]
                yamlfile['type'] = args.type
                with open(myFile, 'w') as outfile:   
                    yaml.dump(yamlfile, outfile,default_flow_style=False, sort_keys=False)

            if args.backend == 'keras':
                print("generating keras files")
                path = '/mlm_templates/cnn_keras.mlm'
                filepath = pkg_resources.resource_filename(__name__, path)
                yamlfile = yaml.load(open(filepath))
                myFile = readfile()
                yamlfile['file'] = myFile
                yamlfile['type'] = args.type
                with open(myFile, 'w') as outfile:   
                    yaml.dump(yamlfile, outfile,default_flow_style=False, sort_keys=False)



        if args.gen == 'ann':
            if args.backend == 'tensorflow2.0':
                print("generating tf files")
                path = '/mlm_templates/ann_tensorflow2.mlm'
                filepath = pkg_resources.resource_filename(__name__, path)
                yamlfile = yaml.load(open(filepath))
                myFile = readfile()
                yamlfile['file'] = myFile.split('.')[0]
                yamlfile['type'] = args.type
                with open(myFile, 'w') as outfile:   
                    yaml.dump(yamlfile, outfile,default_flow_style=False, sort_keys=False)


            if args.backend == 'keras':
                print("generating keras files")
                path = '/mlm_templates/ann_keras.mlm'
                filepath = pkg_resources.resource_filename(__name__, path)
                yamlfile = yaml.load(open(filepath))
                myFile = readfile()
                yamlfile['file'] = myFile.split('.')[0]
                yamlfile['type'] = args.type
                with open(myFile, 'w') as outfile:   
                    yaml.dump(yamlfile, outfile,default_flow_style=False, sort_keys=False)

                

            if args.backend == 'keras':
                print("generating keras files")
                path = '/mlm_templates/ann_keras.mlm'
                filepath = pkg_resources.resource_filename(__name__, path)
                yamlfile = yaml.load(open(filepath))
                myFile = readfile()
                yamlfile['file'] = myFile.split('.')[0]
                yamlfile['type'] = args.type
                with open(myFile, 'w') as outfile:   
                    yaml.dump(yamlfile, outfile,default_flow_style=False, sort_keys=False)


        
        if args.gen == 'lstm': 
            if args.backend == 'keras':
                print("generating keras files")
                path = '/mlm_templates/lstm_keras.mlm'
                filepath = pkg_resources.resource_filename(__name__, path)
                yamlfile = yaml.load(open(filepath))
                myFile = readfile()
                yamlfile['file'] = myFile.split('.')[0]
                print("my file",myFile)
                yamlfile['type'] = args.type
                with open(myFile, 'w') as outfile:   
                    yaml.dump(yamlfile, outfile,default_flow_style=False, sort_keys=False)

            if args.backend == 'tensorflow 2.0':
                print("generating keras files")
                path = '/mlm_templates/lstm_tensorflow2.mlm'
                filepath = pkg_resources.resource_filename(__name__, path)
                yamlfile = yaml.load(open(filepath))
                myFile = readfile()
                yamlfile['file'] = myFile.split('.')[0]
                yamlfile['type'] = args.type
                with open(myFile, 'w') as outfile:   
                    yaml.dump(yamlfile, outfile,default_flow_style=False, sort_keys=False)

    return("")



                
        
if __name__ == '__main__':
    main()