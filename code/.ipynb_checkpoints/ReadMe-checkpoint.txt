File Directories:
input
    - contains the Kaggle dataset downloaded from https://www.kaggle.com/c/understanding_cloud_organization/data
clouds
    - contains the source code for the entire project
parsed_data
    - contains data that is parsed from the cloud image database, as well as the optimal parameters found through model validation 
logs
    - contains all training/validation logs
submissions
    - contains the final submission for each architecture

Steps for downloading the dataset:
1. download the Kaggle api (https://github.com/Kaggle/kaggle-api)
2. cd to the project directory and run <kaggle competitions download -c understanding_cloud_organization>

Steps for selecting the architecture:
1. go to <model.py> and change the following parameters based on your architecture. 
        NETWORK = 'Unet'                 # architecture type
        ENCODER = 'resnet34'             # backbone model
        ENCODER_WEIGHTS = 'imagenet'     # dataset that backbone is trained on

Steps to training the model and returning the final segmentation masks:
1. run <parse_data.py> (parsed dataset stored in /parsed_data)
2. run <train.py> (optimal weights learned and saved in logs)
3. run <validation.py> (optimal parameters calculated and saved in parsed_data/optimal_thresholds)
4. run <test.py> (returns the segmentation masks of the test dataset based on optimal model weights and parameters)
5. the final segmentation masks are found in /submissions
