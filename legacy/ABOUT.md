# Legacy files

These are my original files from the first version of this project, kept unchanged
(`main.py`, `utility.py`, `data_preprocessing.py`, `deep_net.py` and the first `README.md`).
Only the compiled `__pycache__` folder was removed from the repository; it is not source code.

The first version tried to classify clothes images with two outputs at once, a **category**
(4 classes) and a **colour** (3 classes), with a VGG-style network of 13 convolution layers.

Things I notice today, from reading the code:

- It reads a dataset from a hard-coded Windows path on my old computer; the images are not in this
  repository (the README says the data is a modified version of DeepFashion2, whose terms I have not
  checked here, so the new tutorial does not use it).
- `main.py` passes the labels as lists of text and never calls `label_Bin`, so the labels are not
  turned into numbers before training.
- In `deep_net.py`, `build` trains `self.net` and uses `self.batch_size`, but neither is ever
  defined (the model is stored in a local variable called `net`).
- The validation data is built with the *training* labels (`validation_data=(self.testX, train_data)`),
  so the validation curves could not be trusted even if it ran.
- There is no random seed, no data augmentation and no explanation of the results.

The new notebooks in `../notebooks/` rebuild the idea step by step on a public data set,
Fashion-MNIST, with a small network that trains in seconds on a CPU.
