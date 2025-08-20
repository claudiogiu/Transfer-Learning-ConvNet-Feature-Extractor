# Transfer Learning via ConvNet Feature Extraction

## Introduction  

This repository is designed for implementing Transfer Learning using a ConvNet as the feature extractor. The model is trained on a dataset sourced from the research conducted by KOKLU M., UNLERSEN M. F., OZKAN I. A., ASLAN M. F., and SABANCI K. (2022) in their paper *A CNN-SVM study based on selected deep features for grapevine leaves classification* (Measurement, 188, 110425, DOI: [10.1016/j.measurement.2021.110425](https://doi.org/10.1016/j.measurement.2021.110425)).  

The dataset used for training is publicly available at [this link](https://www.muratkoklu.com/datasets/).  

Transfer learning enables models trained on large datasets to be reused for related tasks that have limited data. In the inductive setting, feature extraction helps transfer learned representations between domains with different data distributions, improving adaptability and performance across diverse applications.

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
     - `augmentations.py`: Defines data augmentation strategies for training and transformations for evaluation across data sets.
     - `train_pipeline.py`: Trains a classifier using transfer learning in feature extraction mode and saves the resulting model to the `models/` directory.
     - `evaluate_pipeline.py`: Computes evaluation metrics to assess model performance on the held-out test set. 


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository.  
