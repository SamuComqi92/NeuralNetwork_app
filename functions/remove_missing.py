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

    # Rimozione delle colonne con più del 70% di valori mancanti
    removed_columns = [col for col in dataframe.columns if dataframe[col].isna().mean() > threshold]
    dataframe.drop(columns = removed_columns, inplace = True)

    # Rimozione delle righe con più del 70% di valori mancanti
    row_threshold = int( dataframe.shape[1] * (1 - threshold) )
    dataframe.dropna(thresh = row_threshold, inplace = True)

    # Rest index + colonne finali
    dataframe.reset_index(drop = True, inplace = True)
    selected_columns = dataframe.columns
    
    return dataframe, selected_columns
