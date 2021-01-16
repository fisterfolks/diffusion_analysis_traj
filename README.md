This project part of master's thesis: Methods of Machine learning for anomalous diffusion trajectory recognition.

there are some .ipynb files in folder brownian_motion:
 - classification.ipynb
 This ipynb file contains the results of comparison of methods for solving the problem of classification of different types of anomalous diffusion.
 
 
 Data - sequences of fractional brownian motion that received from generate.py file or generate_fbm_trajectories_dataset.ipynb. All data is one-dimensional(1D)
 
 
 The first task is binary classification - comparison the results of machine learning methods: xgboost, random forest with a neural network with LSTM layers in the presence of a sample of data of two classes (sequences with hurst_coeff = 0.5, 0.85).
 Metric - accuracy. The distribution of two classes is equal.
 
 The second task is multiclassification problem - three classes of sequences anomalous diffusion - superdiffusion (hurst coeff = 0.85), normal diffusion (hurst coeff - 0.5) and subdiffusion (hurst coeff = 0.25).
 
 Metric - accuracy.
 - continuation_of_sequence.ipynb
 Check how neural network with few layers of lstm can predict continuation of sequence. The training data - fbm sequences from generate.py
 
 - generate.py
 File for simulating fractional brownian motion trajectories
 - hurst_variance.py
 Simulating and modelling trajectories of fractional brownian motion(fBm). You can choose different method of simulating, different hurst coefficient, size of sequence, size of dataset and etc for 1D data. Also check msd comparison.
 - generate_fbm_trajectories_dataset.ipynb
 Simulating and modelling trajectories of fractional brownian motion(fBm) - ipynb
 
 
 17\01\21
 
 Task right now 
 - how good neural network for closes trajectories of three classes(0.4, 0.5, 0.7)
 - simulating and modelling fBm 2D
 - simulating and modelling CTRW
 - explanation of results continuation of sequences
 
