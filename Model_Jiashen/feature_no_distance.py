
from feature_engineer import get_engineered_df_allWC


def get_features_without_distance_allWC(file_path, warehouse, max_time=300):
    df, feature_cols, cat_cols = get_engineered_df_allWC(file_path,
                                                         warehouse=warehouse,
                                                         max_time=300)

    df = df.drop(columns=['Prev_Timestamp', 'Prev_LocationID', 'Prev_Aisle',
                          'Prev_Bay', 'Prev_Level', 'Prev_Slot', 'Prev_Aisle2',
                          'Prev_Bay2', 'PrevLocKey', 'Travel_Distance',
                          'same_aisle', 'same_lockey', 'diff_level'])

    to_remove = ['Travel_Distance', 'same_aisle', 'same_lockey', 'diff_level']
    feature_cols = list(filter(lambda x: x not in to_remove, feature_cols))

    to_remove = ['same_aisle', 'same_lockey', 'diff_level']
    cat_cols = list(filter(lambda x: x not in to_remove, cat_cols))

    return df, feature_cols, cat_cols
