import numpy as np
import faiss
GPU = faiss.StandardGpuResources()  # use a single GPU

class Embeddings(object):

    def __init__(self, vectors, identifiers, dim=512, gpu=True, inbuilt_index=False):
        """

        :param vectors:
        :param identifiers:
        :param dim:
        :param gpu:
        :param inbuilt_index:
        """
        Embeddings.validate(vectors,identifiers)
        self.__dimension=dim
        self.__vectors = vectors
        self.__identifiers = identifiers
        self.__gpu = gpu
        self.__inbuilt_index = inbuilt_index
        self.__quantizer = faiss.IndexFlatL2(dim)  # the other index

        if self.__inbuilt_index:
            self.__indexmap = faiss.IndexIDMap2(self.__quantizer)
        else:
            self.__indexmap = self.__quantizer

        if self.__gpu:
            self.__index = faiss.index_cpu_to_gpu(GPU, 0, self.__indexmap)

        else:
            self.__index = self.__indexmap
        self.__add()

    def __add(self,vector=None,identifier=None):
        """

        :param vector:
        :param identifier:
        """
        if vector is None and identifier is None:
            if self.__inbuilt_index:
                self.__index.add_with_ids(self.__vectors, self.__identifiers)
            else:
                self.__index.add(self.__vectors)
        else:
            if self.__inbuilt_index:
                self.__index.add_with_ids(vector,identifier)
            else:
                self.__index.add(vector)

    def add_pair(self,vector, identifier):
        """

        :param vector:
        :param identifier:
        """
        if vector is None or identifier is None:
            raise Exception("NoneValueException")
        Embeddings.validate(vector, identifier)
        self.__vectors = np.append(self.__vectors, vector)
        self.__identifiers = np.append(self.__identifiers,identifier)
        self.__add(vector,identifier)


    def search(self,vector, len=2):
        """

        :param vector:
        :param len:
        :return:
        """
        ret = dict()
        D, I = self.__index.search(vector, len)
        # print(D,I)
        if self.__inbuilt_index:
            ret["identity"] = I[0]
        else:
            ret["identity"] = [self.__identifiers[x] for x in I[0]]
        ret["distance"] = D[0]
        # print(ret)
        return ret

    def remove(self, idx):
        """

        :param idx:
        """
        if self.__gpu:
            raise Exception("UnsupportedMethod")
        self.__index.remove_ids(np.array([idx]))

    @staticmethod
    def validate(vector, idx):
        """

        :param vector:
        :param idx:
        """
        n, d = vector.shape
        assert idx.shape == (n,), 'vectors and identifiers are not of same length'