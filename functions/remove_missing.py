# Importo librerie utili
import pandas as pd

# Funzione per rimuovere le colonne e le righe con più del 70% di valori mancanti
# Tali colonne potenzialmente non hanno nessuna informazione utile
# In altri casi, tali colonne possono essere trasformate per conservare le poche informazioni (NON INCLUSO IN QUESTO SCRIPT)
def remove_missing(dataframe, threshold = 0.7) :
    """
    La funzione accetta come parametro il dataframe da controllare
    La funzione restituisce il dataframe ripulito da righe e colonne con più del 70% di valori mancanti, e la lista di colonne rimaste
    """

    removed_columns = [col for col in dataframe.columns if dataframe[col].isna().mean() > threshold]    # Rimozione delle colonne con più del 70% di valori mancanti
    dataframe.drop(columns = removed_columns, inplace = True)
    row_threshold = int( dataframe.shape[1] * (1 - threshold) )                                         # Rimozione delle righe con più del 70% di valori mancanti
    dataframe.dropna(thresh = row_threshold, inplace = True)
    dataframe.reset_index(drop = True, inplace = True)                                                  # Rest index + colonne finali
    selected_columns = dataframe.columns
    
    return dataframe, selected_columns
