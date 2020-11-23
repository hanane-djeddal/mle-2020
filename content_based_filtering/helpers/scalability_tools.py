def get_subset(dataframe,subcols,id_col):
    """
    dataframe : a pandas dataframe
    subcols : a subset of dataframe.columns including id_col
    id_col : the name of the column containing an ID 
    
    Returns a sorted dataframe (by id_col) containing only the columns in 'subcols' without duplicates in the column 
    'id_col' 
    """
    subdf=dataframe[subcols]
    subdf=subdf.drop_duplicates(subset =id_col, keep = 'first', inplace = False) 
    subdf.sort_values(by=id_col)
    return subdf



