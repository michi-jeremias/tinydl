# Imports
from abc import ABC, abstractmethod
from functools import wraps

import pandas as pd


# Classes
class Augmenter(ABC):

    @abstractmethod
    def assert_datatype(func):
        pass

    @abstractmethod
    def generate(self, data):
        pass


class PandasAugmenter(Augmenter):

    def assert_datatype(func):
        @wraps(func)
        # def wrapper(self, data, colnames):
        def wrapper(self, data, colnames=None):
            DATATYPE = pd.DataFrame
            assert isinstance(
                data, DATATYPE), f"Argument type is not {DATATYPE}"
            return func(data, colnames)
        return wrapper

    @abstractmethod
    def generate(self, data):
        pass


class IsUniqueGenerator(PandasAugmenter):

    @PandasAugmenter.assert_datatype
    def generate(data, colnames=None) -> pd.DataFrame:
        """Returns if a value in a column of a dataframe is
        unique (1) or not (0)

        Example
        -------
        [[1, 0, 0],                 [[0, 0, 0],
         [1, 0, 0],    generates     [0, 0, 0],
         [1, 1, 1],                  [0, 1, 0],
         [2, 0, 1]]                  [1, 0, 0]]

        Parameters
        ----------
        df : pandas.DataFrame
        colnames : List of columns to be checked for unique values

        Returns
        -------
        df_is_unique : pandas.DataFrame
        """
        df_is_unique = pd.DataFrame()

        if colnames is None:
            colnames = data.columns

        for col in colnames:
            count = data[col].value_counts()
            is_unique = {f"{col}_u": data[col].isin(
                count.index[count == 1])}
            df_res = pd.DataFrame.from_dict(is_unique)
            df_is_unique = pd.concat([df_is_unique, df_res], axis=1)

        return df_is_unique * 1


class HasUniqueGenerator(PandasAugmenter):

    @PandasAugmenter.assert_datatype
    def generate(data, colnames=None) -> pd.DataFrame:
        """Returns if a row has at least one value of True or 1 over
        the columns in colnames. It's basically the evaluation of an OR
        statement of a row across all columns.

        Example
        -------
        [[1, 0, 0],                 [[1],
         [0, 0, 0],    generates     [0],
         [1, 1, 1],                  [1],
         [0, 0, 1]]                  [1]]

        Parameters
        ----------
        df : pandas.DataFrame
        colnames : List of columns to be checked for True values

        Returns
        -------
        df_has_unique : pd.DataFrame with a single column of results"""
        if colnames is None:
            colnames = data.columns

        has_unique = {"has_unique": data[colnames].any(axis=1)}
        df_has_unique = pd.DataFrame.from_dict(has_unique)

        return df_has_unique * 1
