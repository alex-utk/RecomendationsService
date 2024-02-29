from typing import List
import torch
import pickle
import numpy as np
from recbole.model.general_recommender.multivae import MultiVAE


class RecboleModel():
    def __init__(self, config_path: str, dataset_path: str, model_path: str, popular_file_path: str) -> None:
        with open(popular_file_path, 'rb') as popular_file:
            self.popular = pickle.load(popular_file)
        with open(config_path, 'rb') as config_file:
            self.config = pickle.load(config_file)
        with open(dataset_path, 'rb') as dataset_file:
            self.dataset = pickle.load(dataset_file)
        self.model = MultiVAE(self.config, self.dataset)
        self.model.load_state_dict(torch.load(model_path)["state_dict"])
        self.model = self.model.cuda()
        self.model.eval()

    def recommend_single(self, user_id: int, k_recos: int = 10) -> List[int]:
        user_id = str(user_id)
        if user_id in self.dataset.field2token_id[self.dataset.uid_field]:
            uid_series = self.dataset.token2id(self.dataset.uid_field, [user_id])  # external tokens to internal id
            index = np.isin(self.dataset[self.dataset.uid_field].numpy(), uid_series)
            new_interactions = self.dataset[index]
            new_interactions = new_interactions.to('cuda:2')
            with torch.inference_mode():
                new_scores = self.model.full_sort_predict(new_interactions)
                new_scores = new_scores.view(-1, self.dataset.item_num)  # reshape
                new_scores[:, 0] = -np.inf
            recommended_item_indices = torch.topk(new_scores, k_recos).indices[0].tolist()
            recos = self.dataset.id2token(self.dataset.iid_field, [recommended_item_indices]).tolist()[0]
            recos = [int(reco) for reco in recos]
        else:
            recos = self.popular
        return recos


class AutoencoderModel():
    def __init__(self, model_path, interactions_path, device='cuda:2') -> List[int]:
        self.model = torch.jit.load(model_path, map_location=device)
        self.interactions_matrix = np.load(interactions_path)
        self.interactions_matrix_torch = torch.Tensor(self.interactions_matrix).to(device)

    def recommend_single(self, user_id, n_random_items=35, n_recos=10):
        input = torch.unsqueeze(self.interactions_matrix_torch[user_id], 0)
        with torch.inference_mode():
            output = self.model(input)
        output = output.numpy(force=True)

        # айтемы, где у нас не нулевое значение
        already_watched = np.argwhere(self.interactions_matrix[user_id] > 0).ravel()
        non_interacted_all = np.setdiff1d(np.arange(self.interactions_matrix[user_id].shape[0]),
                                          already_watched)  # set(all_items) - set(already_watched)
        non_interacted_random_items = np.random.choice(non_interacted_all,
                                                       size=n_random_items,
                                                       replace=False)
        user_preds = output[0][non_interacted_random_items]
        items_idx = non_interacted_random_items[np.argsort(-user_preds)[:n_recos]]

        return items_idx
