import torch


class Profiler:
    def __init__(self, profile_path):
        if profile_path is None:
            self.profiler = None
        else:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True
            )

    def __enter__(self):
        if self.profiler is not None:
            self.profiler.start()
            return self
        else:
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            self.profiler.stop()

    def step(self):
        if self.profiler is not None:
            self.profiler.step()
