from collections import Counter

import pandas as pd
import numpy as np
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender


class UserKnn():
    """Class for fit-perdict UserKNN model, optimised for 1 item recos.
    """

    def __init__(self, model: ItemItemRecommender, popular: list[int], N_users: int = 25):
        self.N_users = N_users
        self.model = model
        self.is_fitted = False
        self.top_n_items = popular

    def get_mappings(self, train):
        self.users_inv_mapping = dict(enumerate(train['user_id'].unique()))  # внутрен -> внешние
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}  # внешние -> внутрен

        self.items_inv_mapping = dict(enumerate(train['item_id'].unique()))  # внутрен -> внешние
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}  # внешние -> внутрен

    def get_matrix(self, df: pd.DataFrame,
                   user_col: str = 'user_id',
                   item_col: str = 'item_id',
                   weight_col: str = None):

        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        self.interaction_matrix = sp.sparse.coo_matrix((
            weights,
            (
                df[item_col].map(self.items_mapping.get),
                df[user_col].map(self.users_mapping.get)
            )
            ))

        self.watched = (df
                        .groupby(user_col, as_index=False)
                        .agg({item_col: list})
                        .rename(columns={user_col: 'sim_user_id'})
                        .set_index('sim_user_id'))

        return self.interaction_matrix

    def idf(self, n: int, x: float):
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame):
        item_cnt = Counter(df['item_id'].values)
        item_idf = pd.DataFrame.from_dict(item_cnt, orient='index',
                                          columns=['doc_freq']).reset_index()
        item_idf['idf'] = item_idf['doc_freq'].apply(lambda x: self.idf(df.shape[0], x))

        self.item_idf = item_idf.set_index('index')['idf']

    def fit(self, train: pd.DataFrame):
        self.get_mappings(train)
        weights_matrix = self.get_matrix(train)
        self._count_item_idf(train)
        self.model.fit(weights_matrix)

        self.is_fitted = True

        self.mapper = self._generate_recs_mapper(
            model=self.model,
            N=self.N_users
        )

    def _generate_recs_mapper(self, model: ItemItemRecommender, N: int):
        def _recs_mapper(user):
            if user not in self.users_mapping:
                return None
            else:
                user_id = self.users_mapping[user]
                users, sim = model.similar_items(user_id, N=N)
                return np.array([self.users_inv_mapping[user] for user in users]), sim
        return _recs_mapper

    def predict(self, test: pd.DataFrame, N_recs: int = 10):
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        recs = pd.DataFrame({'user_id': test['user_id'].unique()})
        recs['sim_user_id'], recs['sim'] = zip(*recs['user_id'].map(self.mapper))
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()

        recs = recs[~(recs['user_id'] == recs['sim_user_id'])]\
            .merge(self.watched, on=['sim_user_id'], how='left')\
            .explode('item_id')\
            .sort_values(['user_id', 'sim'], ascending=False)\
            .drop_duplicates(['user_id', 'item_id'], keep='first')\
            .merge(self.item_idf, left_on='item_id', right_on='index', how='left')

        recs['score'] = recs['sim'] * recs['idf']
        recs = recs.sort_values(['user_id', 'score'], ascending=False)
        recs['rank'] = recs.groupby('user_id').cumcount() + 1
        return recs[recs['rank'] <= N_recs][['user_id', 'item_id', 'score', 'rank']]

    def predict_single(self, user_id: int, N_recs: int = 10) -> list[int]:
        """Предикт, опитмизированный под одного юзера

        Args:
            user_id (int): id юзера
            N_recs (int, optional): сколько нужно рекомендаций. Defaults to 10.

        Raises:
            ValueError: Модель не обучена

        Returns:
            list[int]: n рекомендаций
        """

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        results = self.mapper(user_id)
        if results is None:
            items = self.top_n_items
        else:
            sim_user_ids, similarity_scores = results

            # удаляем самого себя
            mask = ~(sim_user_ids == user_id)
            sim_user_ids = sim_user_ids[mask]
            similarity_scores = similarity_scores[mask]

            similar_items = self.watched.loc[sim_user_ids]  # айтемы похожих людей
            similar_items['scores'] = similarity_scores
            similar_duplicates = similar_items.explode('item_id').to_numpy()

            items = similar_duplicates[:, 0].astype(int)
            scores = similar_duplicates[:, 1].astype(float)
            _, indices = np.unique(items, return_index=True)
            indices = sorted(indices)
            items = items[indices]
            scores = scores[indices]

            scores = scores * self.item_idf.loc[items].to_numpy()

            items = items[np.argsort(scores)[::-1]]  # сортировка по убыванию

        # если получили мало рекомендаций, докинем популярного,
        # что бы всегда было n рекомендаций
        if len(items) < N_recs:
            items = np.concatenate((items, self.top_n_items))

            _, indices = np.unique(items, return_index=True)
            items = items[sorted(indices)]

        return list(items[:N_recs])
