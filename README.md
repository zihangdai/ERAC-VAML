# From Credit Assignment to Entropy Regularization: Two New Algorithms for Neural Sequence Prediction

This is the code we used in our paper
>[From Credit Assignment to Entropy Regularization: Two New Algorithms for Neural Sequence Prediction](arxiv link)

>Zihang Dai\*, Qizhe Xie\*, Eduard Hovy (*: equal contribution)

>ACL 2018

## Requirements

Python 3.6, PyTorch 0.3.0

- CUDA 8: `conda install pytorch=0.3.0 -c pytorch `
- CUDA 9.0: `conda install pytorch=0.3.0 cuda90 -c pytorch `
- CUDA 9.1: `conda install pytorch=0.3.0 cuda91 -c pytorch `

## Reproduce our results on MT (IWSTL 2014)

### Download and preprocess the data

```cd mt && bash preprocess.sh```

### ERAC Training

##### [Step 1]   Pretrain the actor

###### Without input feeding:

```cd erac && python train_actor.py --cuda --work_dir PATH_TO_ACTOR_FOLDER```

###### With input feeding

```cd erac && python train_actor.py --cuda --work_dir PATH_TO_ACTOR_FOLDER —input_feed```

##### [Step 2]   Pretrain the critic

###### Without input feeding

```python train_critic.py --cuda --actor_path PATH_TO_ACTOR_FOLDER/model_best.pt --work_dir PATH_TO_CRITIC_FOLDER```

###### With input feeding

```python train_critic.py --cuda --actor_path PATH_TO_ACTOR_FOLDER/model_best.pt --work_dir PATH_TO_CRITIC_FOLDER --input_feed --tau 0.04```

- `PATH_TO_ACTOR_FOLDER` is the actor folder created in step 1.

##### [Step 3]   Train actor-critic jointly

###### With or without input feeding

```python train_erac.py --cuda --actor_path PATH_TO_ACTOR_FOLDER/model_best.pt --critic_path PATH_TO_CRITIC_FOLDER/model_best.pt```

- `PATH_TO_ACTOR_FOLDER` is the actor  folder created in step 1.
- `PATH_TO_CRITIC_FOLDER` is the critic folder created in step 2.

### VAML Training

##### [Step 1]   Train the Q network

###### Without input feeding

```cd vaml && python train_q.py --cuda --work_dir PATH_TO_QNET_FOLDER```

###### With input feeding

```cd vaml && python train_q.py --cuda --work_dir PATH_TO_QNET_FOLDER --input_feed```

##### [Step 2]   Train the generation model

###### Without input feeding

```python train_vaml.py --cuda --critic_path PATH_TO_QNET_FOLDER/model_best.pt```

###### With input feeding

```python train_vaml.py --cuda --critic_path PATH_TO_QNET_FOLDER/model_best.pt --input_feed```

- `PATH_TO_QNET_FOLDER` is the folder created in step 1.


