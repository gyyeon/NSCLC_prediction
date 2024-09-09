# NSCLC_prediction
Ensemble model for NSCLC recurrence prediction

## Demo
1. Instructions to run on data:

   ex. Single model prediction using max tumor slice

    python train.py --model_type max --cuda 0 --seed 42

   ex. Ensemble model prediction using three slices (bf, max, af)

    python train_NN_3models.py --ensemble_type 3slices --single_type bf max af --cuda 0 --seed 42

2. File organization

    /datasets/results/Single_model/max/Ensemble_model/3slices/
             /model_idx/
                        train_idx.txt
                        test_idx.txt
             /csv/sorted_GESIEMENS_530.csv
             /2D/5mm5slice/max_img/ ~.tif
