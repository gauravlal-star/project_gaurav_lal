# Project_Gaurav_Lal
# Smart Trash Sorter ♻️

This project implements a Deep Learning image classification system to automatically sort garbage into 6 distinct categories: **cardboard, glass, metal, paper, plastic, and trash**. 

It features two models:
1. A custom-built Convolutional Neural Network (CNN) with Spatial Pyramid Pooling (SPP).
2. A fine-tuned **ResNet50** model utilizing transfer learning (which achieved the highest accuracy and is used as our final submitted model).

## Project Structure

```text
Project_Gaurav_Lal/
├── checkpoints/          # Contains the saved model weights (final_weights.pth)
├── data/                  # Sample test images organized by class (10 per class)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── config.py              # Hyperparameters and global settings
├── dataset.py             # PyTorch Dataset, DataLoaders, and Augmentations
├── model.py               # Custom CNN and ResNet50 architectures
├── train.py               # Model training and evaluation loops
├── predict.py             # Batch inference and single-image testing utilities
├── interface.py           # Aliases mapped for automated grading scripts
├── main.py                # End-to-end execution script with CLI arguments
└── README.md              # Installation and execution instructions
```
nstallation Instructions
1. Clone the repository:
Open your terminal or command prompt and run:
```text
Bash
git clone [https://github.com/gauravlal-star/Project_Gaurav_Lal.git](https://github.com/gauravlal-star/Project_Gaurav_Lal.git)
cd Project_Gaurav_Lal
```
2. Create a Virtual Environment:
It is highly recommended to use a virtual environment to prevent library conflicts. Run the following command to create one named venv:
```text
Bash
python -m venv venv
3. Activate the Virtual Environment:

On Mac/Linux:

Bash
source venv/bin/activate
On Windows (Command Prompt):

DOS
venv\Scripts\activate
On Windows (PowerShell):

PowerShell
venv\Scripts\Activate.ps1
(You should now see (venv) at the start of your terminal line).
```
4. Install the Required Dependencies:
With the virtual environment activated, install PyTorch and the required data science libraries using pip:
```text
Bash
pip install torch torchvision
pip install matplotlib pillow scikit-learn seaborn numpy
```
Execution Instructions
This project is highly modular and includes a main.py file with command-line arguments for easy execution. Make sure your virtual environment is activated before running these commands!

1. Running Inference (Testing the Model)
To test the pre-trained ResNet50 model on a single unseen image, use the --predict flag followed by the path to the image.

Example using an image from the provided data folder:
```text
Bash
python main.py --predict data/glass/glass1.jpg
```
(This will output the predicted class and an ASCII confidence bar chart directly in the terminal).ata/glass/glass1.jpg
(This will output the predicted class and an ASCII confidence bar chart directly in the terminal).
