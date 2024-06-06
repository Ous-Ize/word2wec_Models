from abc import abstractmethod

from torch import Tensor


class NegativeSamplingMixin:
    @abstractmethod
    def forward(
        self,
        input_id: Tensor,
        context_ids: Tensor,
        negative_ids: Tensor,
    ) -> Tensor:
        """Perform a negative sampling forward pass."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() is not implemented"
        )
