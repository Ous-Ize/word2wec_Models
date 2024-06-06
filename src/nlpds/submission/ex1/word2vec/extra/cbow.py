from nlpds.abc.ex1.word2vec.extra.cbow import (
    CbowDatasetABC,
    CbowDatasetNegativeSamplingABC,
    CbowNegativeSample,
    CbowNegativeSamplingABC,
    CbowSample,
    CbowSoftMaxABC,
)


class CbowDataset(CbowDatasetABC[CbowSample]):
    def __init__(
        self,
        # ...
    ):
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")


class CbowSoftMax(CbowSoftMaxABC):
    def __init__(
        self,
        # ...
    ):
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")


class CbowNegativeSampling(CbowNegativeSamplingABC):
    def __init__(
        self,
        # ...
    ):
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")


class CbowDatasetNegativeSampling(CbowDatasetNegativeSamplingABC[CbowNegativeSample]):
    def __init__(
        self,
        # ...
    ):
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
