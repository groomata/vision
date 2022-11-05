from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class SimCLRLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
    ):
        """Implements SimCLR's NT-Xent loss.

        Args:
            temperature (float): scaling parameter for softmax.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        representations_1: torch.Tensor,
        representations_2: torch.Tensor,
    ) -> torch.Tensor:

        representations = self._combine_representations(
            representations_1, representations_2
        )

        similarity = self._compare_representations(representations)

        loss = self._evaluate_similarity(similarity)

        return loss

    def _combine_representations(
        self,
        representations_1: torch.Tensor,
        representations_2: torch.Tensor,
    ) -> torch.Tensor:

        representations = rearrange(
            [representations_1, representations_2],
            "g b d -> (b g) d",
        )

        return representations

    def _combine_representations_slow(
        self,
        representations_1: torch.Tensor,
        representations_2: torch.Tensor,
    ) -> torch.Tensor:
        B, D = representations_1.shape

        representations = torch.empty(2 * B, D, dtype=torch.float)

        for i in range(B):
            representations[2 * i] = representations_1[i]
            representations[2 * i + 1] = representations_2[i]

        return representations

    def _compare_representations(
        self,
        representations: torch.Tensor,
    ) -> torch.Tensor:
        representations = F.normalize(representations, dim=1)

        similarity = torch.einsum(
            "i d, j d -> i j",
            representations,
            representations,
        )

        return similarity

    def _compare_representations_slow(
        self,
        representations: torch.Tensor,
    ) -> torch.Tensor:
        N, _ = representations.shape  # N = 2 * batch_size

        similarity = torch.empty(N, N, dtype=torch.float)

        for i in range(N):
            representations[i] = representations[i] / torch.norm(representations[i])

        for i in range(N):
            for j in range(N):
                similarity[i][j] = torch.dot(representations[i], representations[j])

        return similarity

    @staticmethod
    def generate_simclr_positive_indices(
        batch_size: int,
        device: Union[torch.device, str, None] = None,
    ) -> torch.Tensor:
        base = torch.arange(batch_size, device=device)

        odd = base * 2 + 1
        even = base * 2

        return rearrange([odd, even], "g b -> (b g)")

    def _evaluate_similarity(
        self,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        N, _ = similarity.shape

        B = N // 2  # batch_size

        similarity.div_(self.temperature)
        similarity.fill_diagonal_(torch.finfo(similarity.dtype).min)

        loss = F.cross_entropy(
            similarity,
            self.generate_simclr_positive_indices(B, similarity.device),
        )

        return loss

    def _evaluate_similarity_slow(
        self,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        N, _ = similarity.shape

        B = N // 2  # batch_size

        similarity /= self.temperature

        def evaluate_similarity_single(
            anchor_idx: int, positive_idx: int
        ) -> torch.Tensor:
            row = similarity[anchor_idx]
            row[anchor_idx] = torch.finfo(similarity.dtype).min
            probs = row.exp() / row.exp().sum()
            return -probs[positive_idx].log()

        loss = torch.tensor(0.0)
        for i in range(B):
            loss += evaluate_similarity_single(2 * i, 2 * i + 1)
            loss += evaluate_similarity_single(2 * i + 1, 2 * i)

        return loss / N
