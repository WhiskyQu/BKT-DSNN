#!/bin/bash

DATASET='fashion_mnist'  
IMAGE_SIZE=28     
ACTIVATION='relu'  
LOSS_FUNCTION='cross_entropy' 
OPTIMIZER='adam'   
EPOCHS=1          
TIMESTEPS=2      
NUM_CLASS=10     
IN_CHANNELS=1     
BATCH_SIZE=32     
LEARNING_RATE=0.0001  
MODEL_SAVE_PATH='../Save_Model/Fed_train_model/Fashion_mnist_models/Fashion_mnist'  
LOG_SAVE_PATH='../Save_Logs/Fed_train_logs/Fashion_mnist.log' 
SAVE_FREQUENCY=5 
MODEL_NAME="ANN_VGG11" 


CLIENTS=5        
ROUNDS=20          
LOCAL_EPOCHS=2    
CLIENT_SELECTION_RATIO=0.5  
PRUNE_RATION=0.5  

python ../Fed_train_with_distillation_GMBS_KTL.py \
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

