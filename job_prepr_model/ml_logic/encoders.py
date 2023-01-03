from tensorflow.keras.utils import to_categorical
from job_prepr_model.ml_logic.params import y_label_dict
import numpy as np
import pandas as pd

def label_encode(y):

    y_df = pd.DataFrame(y)
    y_df_num = y_df.applymap(lambda x : y_label_dict[x])
    y_num = np.array(y_df_num)

    y_cat = to_categorical(y_num)

    return y_cat
