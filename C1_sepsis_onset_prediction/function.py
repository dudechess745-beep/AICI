# Provide file handling, serialization, and data utility functions.
import os
import pickle
import pandas as pd
import time

def save_pickle(obj, filename, use_dill=False, protocol=4):
    _create_folder_if_not_exist(filename)
    with open(filename, 'wb') as file:
        if not use_dill:
            pickle.dump(obj, file, protocol=protocol)
        else:
            import dill
            dill.dump(obj, file)

def load_pickle(filename, use_dill=False):
    with open(filename, 'rb') as file:
        if not use_dill:
            obj = pickle.load(file)
        else:
            import dill
            obj = dill.load(file)
    return obj

def _create_folder_if_not_exist(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

def numpy_to_named_dataframe(data, idx, name, last_only=False):
    col_names = [name + '_{}'.format(i) for i in range(1, data.shape[-1] + 1)]
    if last_only is False:
        if isinstance(data, pd.DataFrame):
            data = data.values
        return pd.DataFrame(index=idx, data=data, columns=col_names)
    else:
        if isinstance(data, pd.DataFrame):
            data = data.values
        return pd.DataFrame(index=idx, data=data[[-1], :], columns=col_names)

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r \n  %2.2f ms' % (method, (te - ts) * 1000))
        return result
    return timed
