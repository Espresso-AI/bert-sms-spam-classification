# Acc : 99.61 | ham F1 score : 99.77 | spam F1 score : 98.65 for SMS Spam Collection Dataset  

This repo is the original code of a [Kaggle notebook](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) for [SMS Spam Collection Dataset](https://www.kaggle.com/code/hotcouscous/acc-99-61-ham-spam-f1-score-99-77-98-65).  

What you can additionally do on this code;  
1. manage arguments, callbacks, and other various features of pl.Trainer with a single yaml file
2. configure all the hyperparameters of an experiment with a single yaml file
3. check the past config through logs of hydra 
4. trace model training by wandb
  
## Result
||F1 score|Accuracy| 
|---|---|---|
|ham|99.77|-|
|spam|98.65|-|
|total|99.61|99.61|

The model is tested on 517 samples selected as 10% of the entire dataset. The validation dataset is randomly sampled each time the training runs and it does not involve in the model's learning.  

As an objective loss, I adopted focal loss to deal with positive/negative imbalance. Instead of following the paper, I implemented it in the form of multi-class classification, which makes training more stable. In addition, for better stability of training, I adopted regularization by AdamW, warmup start, and linear-decreasing lr scheduler.    

## Usage
To train the model on this repo, run the command line;  
```
python train.py —config-name exp_0
```

If you want to test the checkpoints, enter checkpoints file path on exp_0.yaml and run the command line;  
```
python test.py —config-name exp_0
```

## License
BSD 3-Clause License Copyright (c) 2022
