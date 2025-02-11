# Mail Message Spam Detection
**Authors :**
- [Salaheddine Bahlaouane](https://github.com/Skeiba)
- [Yassin Zaher](https://github.com/Yassin-Zaher)

Welcome to the **Mail Message Spam Detection** project ! This repository contains a Python-based implementation for detecting spam in email messages. It includes both the code for model training and a Flask web application to interact with the model.

---

## Project Structure

```
├ app/                 		        # Flask web application
   ├── static/          	        # Static files
   ├── model/           	        # Pre-trained model and vectorizer files
   ├── templates/                   # HTML template
   ├── app.py               	    # Flask app   
   ├── enron_spam_data              # Enron spam dataset
   ├── spam_detection_model.ipynb   # Jupyter notebook with model training and analysis
   └── README.md            	    # Project documentation
```

---

## Features

* **Machine Learning Model** : A trained spam detection model using natural language processing (NLP) techniques.
* **Web Application** : A user-friendly interface to input email messages and predict whether they are spam or ham.
* **Interactive Notebook** : A detailed walkthrough of the model development process.

---

## Getting Started

### Prerequisites

Python 3.8+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Skeiba/spamDetectionPython.git
   cd mail-message-spam-detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python -m flask run
   ```
5. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## Usage

1. **Train the Model** : Use the `spam_detection_model.ipynb` notebook to understand and retrain the model if necessary.
2. **Run the App** : Use the Flask app to input email messages and classify them as spam or ham.

---

## Model Details

The spam detection model leverages NLP techniques, such as:

* Tokenization
* Text preprocessing (e.g., stop-word removal, stemming)
* TF-IDF vectorization

It is trained using a machine learning algorithm Random Forest on a labeled dataset of spam and ham messages.

---

## Acknowledgments

* Datasets and inspiration from open-source spam classification datasets.
* Flask for the web framework.
* Scikit-learn and other Python libraries for model training.