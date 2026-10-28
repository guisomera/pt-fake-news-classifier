from sklearn.model_selection import train_test_split

from load_data import load_data


def get_split():
    table = load_data()
    train_ids, test_ids = train_test_split(table["id"].unique(), test_size=0.2, random_state=42)
    train = table[table["id"].isin(train_ids)]
    test = table[table["id"].isin(test_ids)]
    return train, test