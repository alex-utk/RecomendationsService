import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from models.neural_nets import AutoencoderModel, RecboleModel

WEIGHTS_PATH = "./models/weights"


class BaseModel(ABC):
    @abstractmethod
    def get_reco(self, user_id: int, k_recs: int):
        pass


class DummyModel(BaseModel):
    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        return list(range(k_recs))


class AnnModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        with open(os.path.join(WEIGHTS_PATH, "ann_model.pkl"), "rb") as file:
            self.model = pickle.load(file)

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        recos = self.model.recommend_single(user_id, k_recs)
        return list(recos)


class UserKNNModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        with open(os.path.join(WEIGHTS_PATH, "user_knn.pkl"), "rb") as file:
            self.model = pickle.load(file)

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        recos = self.model.predict_single(user_id, k_recs)
        return recos


class PrecomputeModel(BaseModel):
    """Класс для заранее просчитанных результатов"""

    def __init__(self, model_path: str, popular_file_path: str) -> None:
        super().__init__()
        with open(model_path, "rb") as hot_file:
            self.hot_recos = pickle.load(hot_file)
        with open(popular_file_path, "rb") as popular_file:
            self.popular = pickle.load(popular_file)

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        return self.hot_recos[user_id] if user_id in self.hot_recos else self.popular


class AutoEncoderWrapper(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = AutoencoderModel("autoencoder.torchscript", "interactions_matrix.npy")

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        recos = self.model.recommend_single(user_id, n_recos=k_recs)
        return recos


class RecboleWrapper(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = RecboleModel(
            "config.pkl", "dataset.pkl", "./saved/MultiVAE-Feb-18-2024_19-09-29.pth", "popular.pkl"
        )

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        recos = self.model.recommend_single(user_id, k_recos=k_recs)
        return recos


def get_models(is_test: bool) -> Dict[Any, Any]:
    if is_test:
        return {"dummy_model": DummyModel()}
    return {
        "ann_model": AnnModel(),
        "userknn_always10_TRUE": UserKNNModel(),
        "dssm_v1": PrecomputeModel("dssm_recos.pkl", "popular.pkl"),
        "autoencoder_v1": AutoEncoderWrapper(),
        "recbole_vae_epoch": RecboleWrapper()
    }
