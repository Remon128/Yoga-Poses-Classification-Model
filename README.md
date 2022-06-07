# Yoga-Poses-Classification-Model
This project works on classifying different yoga poses which help in performing the correct yoga actions and classifies it into one of many yoga poses. the main idea of project is enhancing an existed kaggle model to give better accuracies.

### Used Libraries

```
import os 
import pandas as pd
import numpy as np
import tensorflow as tf
```

## Data Sample

![__results___7_0](https://user-images.githubusercontent.com/24530726/172459333-9897c92b-2bbf-41f3-8031-701ed3cca317.png)


### Enhanced Model
```
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense, Conv2D
# covnet
model = tf.keras.models.Sequential([Conv2D(128,(3,3),input_shape=(224,224,3),activation='relu'),
                                   BatchNormalization(),
                                   Conv2D(128,(3,3)),
                                   BatchNormalization(),
                                   MaxPool2D(2,2),
                                   Conv2D(64,(3,3)),
                                   BatchNormalization(),
                                   Conv2D(64,(3,3)),
                                   BatchNormalization(),
                                   MaxPool2D(2,2),
                                   Conv2D(32,(3,3)),
                                   BatchNormalization(),
                                   Conv2D(32,(3,3)),
                                   BatchNormalization(),
                                   MaxPool2D(2,2),
                                   Flatten(),
                                   Dense(1024,activation='relu'),
                                   BatchNormalization(),
                                   Dense(512,activation='relu'),
                                   Dense(107,activation='softmax')])
```
## CNN Model Base Architecture

![CNN Model](https://user-images.githubusercontent.com/24530726/172461568-80639e2a-abe8-42ac-841b-90b737dd35fa.png)


### Old CNN Model

```
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense, Conv2D
# covnet
model = tf.keras.models.Sequential([Conv2D(128,(3,3),input_shape=(224,224,3),activation='relu'),
                                   Conv2D(128,(3,3)),
                                   MaxPool2D(2,2),
                                   Conv2D(64,(3,3)),
                                   Conv2D(64,(3,3)),
                                   MaxPool2D(2,2),
                                   Conv2D(32,(3,3)),
                                   Conv2D(32,(3,3)),
                                   MaxPool2D(2,2),
                                   Flatten(),
                                   Dense(1024,activation='relu'),
                                   Dense(512,activation='relu'),
```


## Conclusion

### Old results

![Old_Acc](https://user-images.githubusercontent.com/24530726/172462046-49eb3b87-4d00-4486-a013-d812ae8c8d9b.png)


### New results
![New_Acc](https://user-images.githubusercontent.com/24530726/172462098-93c80aa5-d120-4b08-af67-b613b6703bc7.png)
