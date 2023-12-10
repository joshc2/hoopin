import pkg_resources
import pandas as pd

def load_data(name = 'data'):
    """
    Function to load clean or raw data from package.
    Data is read in as a pandas DataFrame.
    Parameters
    ----------
    name : string
    Options are 'data', and 'data_raw'

    Returns
    -------
    A pandas dataframe of the requested data

    """

    # if name == 'data':
    path = 'data/basketball.csv'

    # else:
    #     raise NameError("{} is not recognized. The only names are 'data' and 'data_raw'.".format(name))
    

    data_path = pkg_resources.resource_filename('hoopin', path)
    
    # data.to_csv("data/basketball", index=False)
    return pd.read_csv(data_path)


