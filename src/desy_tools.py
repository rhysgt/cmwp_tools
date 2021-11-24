import numpy as np
import pandas as pd

def load_fio(filename): 
    """ Parse a fio file.
    
    Params
    -------
    filename: str
        Path to fio file.
        
    Returns
    ---------
    df: pd.DataFrame
        Dataframe containing data from fio file.
    
    """
    # Read in column names and data type
    colnames = []; formats = [];
    with open(filename) as input_data:
        for i, line in enumerate(input_data):
            if ' Col' in line:
                colnames.append(' '.join(line.split(' ')[3:-1]))
                skip = i+1
                if 'DOUBLE' in line.split(' ')[-1]: formats.append('f4')
                if 'INTEGER' in line.split(' ')[-1]: formats.append('i4')
                if 'STRING' in line.split(' ')[-1]: formats.append('str')

    # Read in log file into dataframe and remove clearing frames
    df = pd.read_csv(filename, names = colnames, skiprows=skip, sep=' ', skipinitialspace=True) 
    df = df[df.type != 'clearing']

    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    return df
