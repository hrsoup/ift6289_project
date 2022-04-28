# ift6289_project

This is the final project for ift6289 at UdeM.

## Installation

We use a single GPU (NVIDIA GeForce 2080 Ti) to develop this system with:
- Anaconda 3 (python 3.7)
- torch 1.10.2+cu113
- sklearn
- nltk 3.7
- pypinyin 0.46.0
- jieba 0.42.1 

## Usage

(1.1) Using mixed poetry data and music data to train, and testing on music data.

```bash
python main_poetry.py train character
python main_poetry.py test character
```
(2) Using mixed tones data and music data to train, and testing on music data.

```bash
python main_poetry.py train tone
python main_poetry.py test tone
```

(2) Using mixed English jokes data and music data to train, and testing on music data.

```bash
python main_joke.py train joke
python main_joke.py test joke

```
(4) Only using language data to train.

```bash
python main_baseline.py train_test 
```

## Results

The original experimental results of our project are in the `experiment_data.txt`.
