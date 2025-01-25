import logging
import time
import torch
import inspect

def measure_cuda_usage(device=torch.device("cuda:0")):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Measure start time and memory
            start_time = time.time()
            start_allocated = torch.cuda.memory_allocated(device)
            start_reserved = torch.cuda.memory_reserved(device)

            # Execute the function
            result = func(*args, **kwargs)

            # Measure end time and memory
            end_time = time.time()
            end_allocated = torch.cuda.memory_allocated(device)
            end_reserved = torch.cuda.memory_reserved(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)

            # Get caller's file name and line number
            caller_frame = inspect.stack()[1]  # Caller of the decorated function
            caller_filename = caller_frame.filename
            caller_lineno = caller_frame.lineno

            # Log memory and time usage with caller info
            logging.info(
                f"File: {caller_filename}, Line: {caller_lineno}\n"
                f"Allocated before: {start_allocated/1e6:.2f} MB\n"
                f"Allocated after:  {end_allocated/1e6:.2f} MB\n"
                f"Net allocated change:  {(end_allocated - start_allocated)/1e6:.2f} MB\n"
                f"Reserved before:  {start_reserved/1e6:.2f} MB\n"
                f"Reserved after:   {end_reserved/1e6:.2f} MB\n"
                f"Net reserved change:   {(end_reserved - start_reserved)/1e6:.2f} MB\n"
                f"Peak allocated:         {peak_allocated/1e6:.2f} MB\n"
                f"Runtime: {end_time - start_time:.2f} seconds"
            )

            return result
        return wrapper
    return decorator

class MeasureCudaUsage:
    def __init__(self, device=torch.device("cuda:0")):
        self.device = device

    def __enter__(self):
        self.start_time = time.time()
        self.start_allocated = torch.cuda.memory_allocated(self.device)
        self.start_reserved = torch.cuda.memory_reserved(self.device)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Measure end time and memory
        end_time = time.time()
        end_allocated = torch.cuda.memory_allocated(self.device)
        end_reserved = torch.cuda.memory_reserved(self.device)
        peak_allocated = torch.cuda.max_memory_allocated(self.device)

        # Get caller's file name and line number
        caller_frame = inspect.stack()[1]  # 1 frame above this method
        caller_filename = caller_frame.filename
        caller_lineno = caller_frame.lineno

        # Log memory and time usage with caller info
        logging.info(
            f"File: {caller_filename}, Line: {caller_lineno}\n"
            f"Allocated before block: {self.start_allocated/1e6:.2f} MB\n"
            f"Allocated after block:  {end_allocated/1e6:.2f} MB\n"
            f"Net allocated change:  {(end_allocated - self.start_allocated)/1e6:.2f} MB\n"
            f"Reserved before block:  {self.start_reserved/1e6:.2f} MB\n"
            f"Reserved after block:   {end_reserved/1e6:.2f} MB\n"
            f"Net reserved change:  {(end_reserved - self.start_reserved)/1e6:.2f} MB\n"
            f"Peak allocated:         {peak_allocated/1e6:.2f} MB\n"
            f"Runtime: {end_time - self.start_time:.2f} seconds"
        )