import pandas as pd

def prepare_class_weights():
    artists = pd.read_csv('artists.csv')
    artists.shape

    # Sort artists by number of paintings
    artists = artists.sort_values(by=['paintings'], ascending=False)

    artists_top = artists[artists['paintings'] >= 50].reset_index()
    artists_top = artists_top[['name', 'paintings']]
    artists_top['class_weight'] = max(artists_top.paintings)/artists_top.paintings
    artists_top

    class_weights = artists_top['class_weight'].to_dict()
    return class_weights