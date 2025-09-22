🌟 MNIST Digit Classification
=============================

This project focuses on classifying handwritten digits from the MNIST dataset using machine learning. A Random Forest classifier is applied to a processed version of the dataset, where each grayscale digit image is represented as a flattened vector of 784 features (28×28 pixels). The objective is to build a robust model capable of recognizing digits (0--9) with high accuracy.

The project follows a complete data science pipeline: starting with exploratory data analysis (EDA) to understand the dataset, applying preprocessing to prepare the data for training, and then implementing machine learning models. Random Forest is used as the primary algorithm due to its ability to handle high-dimensional data and reduce overfitting compared to single decision trees.

The analysis also evaluates the model with quantitative and visual metrics, such as accuracy scores and confusion matrices, providing an in-depth view of classification performance across digit classes.

* * * * *

✨ Key Features & Technical Details
----------------------------------

-   **Exploratory Data Analysis (EDA):**

    -   The dataset is inspected for its structure and balance.

    -   A distribution plot shows the count of each digit, ensuring that the dataset covers all classes fairly evenly.

-   **Data Preprocessing:**

    -   Pixel values are high-dimensional and vary in scale. To standardize them, `StandardScaler` is applied, ensuring the features contribute equally to the learning process.

-   **Random Forest Classifier:**

    -   A `RandomForestClassifier` is trained to classify digits.

    -   Random Forest leverages an ensemble of decision trees, reducing variance and improving predictive performance.

    -   The model captures non-linear relationships in pixel intensities, making it well-suited for digit recognition.

-   **Evaluation Metrics:**

    -   Overall accuracy provides a clear measure of model effectiveness.

    -   A confusion matrix is generated, which reveals how well the classifier performs on each digit class. Misclassifications (e.g., between similar digits like 4 and 9) are highlighted to understand weaknesses.

* * * * *

🚀 Getting Started
------------------

To run this project, you will need a Python environment with the following libraries:

-   numpy

-   pandas

-   matplotlib

-   seaborn

-   scikit-learn

Steps to reproduce the analysis:

1.  Clone the repository or download the notebook.

2.  Place the `MNIST_Shortened.csv` dataset in the same directory as the notebook.

3.  Open and run the `MNIST Analysis.ipynb` file in Jupyter Notebook or JupyterLab.

* * * * *

📊 Project Workflow
-------------------

The `MNIST Analysis.ipynb` notebook is organized into the following stages:

1.  **Data Loading & Inspection:**

    -   Imports required libraries.

    -   Loads the dataset using `numpy.genfromtxt`.

    -   Splits the data into features (`X`) and labels (`y`).

    -   Checks dataset dimensions to confirm 784 features per image.

2.  **Data Exploration:**

    -   Uses seaborn's `countplot` to visualize digit distribution.

    -   Confirms that all digits (0--9) are represented and checks for class balance.

3.  **Preprocessing:**

    -   Applies `StandardScaler` to normalize pixel intensity values.

    -   Ensures all input features fall within a comparable range, improving model training stability.

4.  **Model Building:**

    -   Trains a `RandomForestClassifier` on the standardized dataset.

    -   Random Forest parameters (number of trees, depth) are tuned to balance performance and computation.

5.  **Model Evaluation:**

    -   Evaluates predictions using `accuracy_score` for overall performance.

    -   Generates a confusion matrix to visualize per-class results, identifying common misclassifications such as confusions between visually similar digits.

* * * * *

📈 Final Thoughts
-----------------

This project demonstrates the strength of ensemble methods like Random Forest for image classification tasks. Even without deep learning, the classifier achieves strong results on digit recognition, making it a solid baseline for comparison.

Future work could explore:

-   Implementing additional algorithms such as Support Vector Machines (SVM), Gradient Boosting, or Neural Networks.

-   Testing convolutional neural networks (CNNs), which are highly effective for image data.

-   Expanding hyperparameter tuning to optimize Random Forest performance further.

* * * * *

🙏 Acknowledgments
------------------

Special thanks to the developers of **numpy**, **pandas**, **matplotlib**, **seaborn**, and **scikit-learn**, whose libraries made this analysis possible. The MNIST dataset, created by Yann LeCun and colleagues, remains an essential benchmark in the field of machine learning and computer vision.
