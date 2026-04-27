# Project_Gaurav_Lal
# Smart Trash Sorter ♻️

This project implements a Deep Learning image classification system to automatically sort garbage into 6 distinct categories: **cardboard, glass, metal, paper, plastic, and trash**. 

It features two models:
1. A custom-built Convolutional Neural Network (CNN) with Spatial Pyramid Pooling (SPP).
2. A fine-tuned **ResNet50** model utilizing transfer learning (which achieved the highest accuracy and is used as our final submitted model).

## 📂 Project Structure

```text
Project_Gaurav_Lal/
├── _checkpoints/          # Contains the saved model weights (final_weights.pth)
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
Installation Instructions
1. Clone the repository:
Open your terminal or command prompt and run:

Bash
git clone [https://github.com/gauravlal-star/Project_Gaurav_Lal.git](https://github.com/gauravlal-star/Project_Gaurav_Lal.git)
cd Project_Gaurav_Lal
2. Install the required dependencies:
Ensure you have Python 3.8+ installed. Then, install the required PyTorch and data science libraries using pip:

Bash
pip install torch torchvision
pip install matplotlib pillow scikit-learn seaborn numpy
🚀 Execution Instructions
This project is highly modular and includes a main.py file with command-line arguments for easy execution.

1. Running Inference (Testing the Model)
To test the pre-trained ResNet50 model on a single unseen image, use the --predict flag followed by the path to the image.

Example using an image from the provided data folder:

Bash
python main.py --predict data/glass/glass1.jpg
(This will output the predicted class and an ASCII confidence bar chart directly in the terminal).

2. Training the Models
If you wish to train the models from scratch on your own machine, you must first replace the small ./data/ folder with the full TrashNet dataset. Then, run one of the following commands:

Bash
# Train both the Custom CNN and ResNet50 models sequentially
python main.py 

# Train ONLY the ResNet50 model
python main.py --model resnet50

# Train ONLY the Custom CNN model
python main.py --model custom
