# NSCLC_prediction
Ensemble model for NSCLC recurrence prediction

## Demo:
(1) Instructions to run on data:
Ex. Single model prediction using max tumor slice

    python train.py --model_type max --cuda 0 --seed 42

Ex. Ensemble model prediction using three slices (bf,max,af)

    python train_NN_3models.py --ensemble_type 3slices --single_type bf max af --cuda 0 --seed 42

(2)  
