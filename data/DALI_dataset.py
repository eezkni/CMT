import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.ops import decoders


class DALIPngLoader(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_root, img_size):
        super().__init__(batch_size, num_threads, device_id)

        self.input = ops.FileReader(file_root=data_root, random_shuffle=False)
