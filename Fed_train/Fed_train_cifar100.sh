#!/bin/bash

DATASET='cifar100'  
IMAGE_SIZE=32    
ACTIVATION='relu'  
LOSS_FUNCTION='cross_entropy'  
OPTIMIZER='adam'  
#EPOCHS=1          
TIMESTEPS=2       
NUM_CLASS=100     
IN_CHANNELS=3    
BATCH_SIZE=32     
LEARNING_RATE=0.001 
MODEL_SAVE_PATH='../Save_Model/Fed_train_model/Fed_cifar100_models/Fed_cifar100' 
LOG_SAVE_PATH='../Save_Logs/Fed_train_logs/Fed_cifar100.log'  
SAVE_FREQUENCY=5 
MODEL_NAME="SNN_VGG11_BNTT"


CLIENTS=2         
ROUNDS=100         
LOCAL_EPOCHS=2    
CLIENT_SELECTION_RATIO=0.5  
PRUNE_RATION=0.3  

python ../Fed_train.py \
  --dataset $DATASET \
  --image_size $IMAGE_SIZE \
  --activation $ACTIVATION \
  --loss_function $LOSS_FUNCTION \
  --optimizer $OPTIMIZER \
  --epochs $EPOCHS \
  --timesteps $TIMESTEPS \
  --num_class $NUM_CLASS \
  --in_channels $IN_CHANNELS \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --model_save_path $MODEL_SAVE_PATH \
  --log_save_path $LOG_SAVE_PATH \
  --save_frequency $SAVE_FREQUENCY \
  --model_name $MODEL_NAME \
  --clients $CLIENTS \
  --rounds $ROUNDS \
  --local_epochs $LOCAL_EPOCHS \
  --client_selection_ratio $CLIENT_SELECTION_RATIO \
  --prune_ratio $PRUNE_RATION \
