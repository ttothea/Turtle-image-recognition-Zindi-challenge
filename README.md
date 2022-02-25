# Capstone Project - [Turtle Recall: Conservation Challenge](https://zindi.africa/competitions/turtle-recall-conservation-challenge) 

Team : Maike, Tobias, Tolga, Kai-Yang

---
Vorschlag
Change repo name to : 2022/02: Turtle Recall  Image Recognition (Zindi Challenge / Neue Fische Capstone)
---


## Description
Being able to distinguish between individuals of the same species is a critical tool for modern conservation. For example, sea turtle conservation efforts aim to track individual turtles to help reveal patterns of movement and residency. Sea turtles are a known indicator species which means that their presence and abundance reflects the health of the wider ecosystem. Therefore, increasing our ability to identify and understand them can enhance our ecological understanding.

Sea turtles can be identified using their facial scales, which are as unique as a human fingerprint. Traditionally, individual recognition has been achieved manually through the attachment of tags on the flippers of found individuals. However, ecological investigations that require recapturing the individuals to study changes in the population dynamics are severely affected by the loss of the tags (or of the marks on the tags). Moreover, tags are expensive and through the long duration of sea turtle life cycles they can deteriorate and require a replacement.

The aim of this competition is to build a machine learning model to identify individual sea turtles. For each image presented, the model should output the turtle’s unique ID or, if the image corresponds to a new turtle (not present in the database), identify it as a new individual. A dataset of labelled photos of turtle faces and a tutorial including a simple baseline model have been provided.


## Environment
Make sure you have the latest version of macOS (currently Monterey) installed.
Also make sure that xcode is installed and updated: 

```BASH
xcode-select --install
```

Then we can go on to install hdf5:

```BASH
 brew install hdf5
```
With the system setup like that, we can go and create our environment and install tensorflow

```BASH
pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.1

pip install -U pip
pip install --no-binary=h5py h5py
pip install tensorflow-macos
pip install tensorflow-metal
pip install -r requirements.txt
```
