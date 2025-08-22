# Transfer Learning via ConvNet Feature Extraction

## Introduction  

This repository is designed for implementing Transfer Learning via a ConvNet in feature extraction mode. The model is trained on a dataset sourced from the research conducted by KOKLU M., UNLERSEN M. F., OZKAN I. A., ASLAN M. F., and SABANCI K. (2022) in their paper *A CNN-SVM study based on selected deep features for grapevine leaves classification* (Measurement, 188, 110425, DOI: [10.1016/j.measurement.2021.110425](https://doi.org/10.1016/j.measurement.2021.110425)).  

The dataset used for training is publicly available at [this link](https://www.muratkoklu.com/datasets/).  

Transfer Learning enables the reuse of models trained on large-scale datasets to address related tasks with limited labeled data. Within the inductive paradigm, in which source and target tasks differ while sharing the same input space, feature extraction is performed by maintaining the weights of a pre-trained architecture unchanged and introducing a task-specific output layer to refine the learned representations to the domain of interest.

## Getting Started 

To set up the repository properly, follow these steps:  

**1.** **Create the Data Directory**  
   - Before running the pipeline, create a `data/` folder in the project root.  
   - Inside `data/`, create two subdirectories:  
     - `raw/`: This will store the unprocessed dataset.  
     - `processed/`: The data will be split into **training and test sets** and saved here.
  
**2. Set Up the Python Environment**  
 
   - Create and activate a virtual environment:  

     ```sh
     python3 -m venv venv
     source venv/bin/activate  # On Windows use: venv\Scripts\activate 
     ```

   - Install dependencies from `requirements.txt`:  

     ```sh
     pip install -r requirements.txt 
     ``` 

**3. Execute the Pipeline with Makefile**  
   - The repository includes a **Makefile** to automate execution of scripts in the `src/` folder.  
   - Run the following command to execute the full workflow:  

     ```sh
     make run_all  
     ```  
   
   - This command sequentially runs the following modules:
     - `preprocess.py`: Splits the raw image dataset into training and test sets.
     - `augmentations.py`: Defines data augmentation strategies for training and transformations for evaluation over data sets.
     - `train_model.py`: Trains a model using Transfer Learning based on a ConvNet in feature extraction mode and saves the trained model to the `models/` directory.
     - `evaluate_model.py`: Computes metrics to validate the model's performance. 


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository.  
