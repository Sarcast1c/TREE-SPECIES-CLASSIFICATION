# 🌲 Tree Species Classification Project

This project aims to classify tree species based on leaf or tree images using various data preprocessing and cleaning techniques. It focuses heavily on dataset preparation and quality assurance before training any deep learning or machine learning models.

---

## 📦 Dataset Overview

The dataset is cloned from the [TREE-SPECIES-CLASSIFICATION](https://github.com/Sarcast1c/TREE-SPECIES-CLASSIFICATION) repository and contains labeled folders for different tree species.

- Each folder represents a unique tree species.
- Images are expected to be in `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.gif` formats.

---

## 📁 Project Structure

├── AICTE_PROJECT.ipynb # Main notebook with data analysis and cleaning

├── TREE-SPECIES-CLASSIFICATION/

└── Tree_Species_Dataset/ # Contains subfolders for each tree species

├── README.md # Project documentation


---

## 📌 Objectives

- Explore the dataset for tree species.
- Identify the number of classes (species).
- Count images per class.
- Visualize sample images from each species.
- Analyze image dimensions (width, height).
- Detect and report:
  - Duplicate images
  - Corrupt/damaged images

---

## 🛠️ Technologies Used

- Python 3.13
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- OS and Hashlib for file management

---

## 🧪 Processing and Cleaning Workflow

1. **Clone dataset from GitHub**
2. **Load class directories** to identify species
3. **Visualize sample images** per class
4. **Analyze image size distribution**
5. **Detect duplicate images** using MD5 hash comparison
6. **Check for corrupted images** using PIL
7. (Optional Future Step) Train classification models using CNNs

---

## 📈 Sample Insights

- Total number of classes (species): _N_
- Class distribution: Bar chart with top species shown
- Common image size (WxH): Reported through `describe()` of DataFrame
- Number of duplicate image sets: _N_
- Number of corrupted images found: _N_

---

## 🚀 Future Plans

- Resize and normalize all images
- Balance dataset across species
- Train using CNNs (e.g., MobileNet, ResNet, EfficientNet)
- Build a web app to upload and classify images

---

## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/tree-species-classifier.git
   cd tree-species-classifier

