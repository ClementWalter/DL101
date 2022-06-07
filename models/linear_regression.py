from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def LinearRegression(**kwargs):
    """
    Standard linear regression model passing all kwargs to a single Dense layer
    """
    return Sequential([Dense(**kwargs)])
