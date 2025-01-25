import logging
import time
import torch

def measure_resource_usage(device=torch.device("cuda:0")):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Measure start time and memory
            start_time = time.time()
            start_allocated = torch.cuda.memory_allocated(device)
            start_reserved = torch.cuda.memory_reserved(device)
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Measure end time and memory
            end_allocated = torch.cuda.memory_allocated(device)
            end_reserved = torch.cuda.memory_reserved(device)
            end_time = time.time()
            peak_allocated = torch.cuda.max_memory_allocated(device)
            
            # Log memory and time usage
            logging.info(
                f"\nAllocated before: {start_allocated/1e6:.2f} MB\n"
                f"Allocated after:  {end_allocated/1e6:.2f} MB\n"
                f"Net allocated change:  {(end_allocated - start_allocated)/1e6:.2f} MB\n"
                f"Reserved before:  {start_reserved/1e6:.2f} MB\n"
                f"Reserved after:   {end_reserved/1e6:.2f} MB\n"
                f"Net reserved change:  {(end_reserved - start_reserved)/1e6:.2f} MB\n"
                f"Peak allocated:        {peak_allocated/1e6:.2f} MB\n"
                f"Runtime: {end_time - start_time:.2f} seconds"
            )
            return result
        return wrapper
    return decorator

class MeasureResourceUsage:
    def __init__(self, device=torch.device("cuda:0")):
        self.device = device

    def __enter__(self):
        # Measure start time and memory
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

        # Log memory and time usage
        logging.info(
            f"\nAllocated before block: {self.start_allocated/1e6:.2f} MB\n"
            f"Allocated after block:  {end_allocated/1e6:.2f} MB\n"
            f"Net allocated change:  {(end_allocated - self.start_allocated)/1e6:.2f} MB\n"
            f"Reserved before block:  {self.start_reserved/1e6:.2f} MB\n"
            f"Reserved after block:   {end_reserved/1e6:.2f} MB\n"
            f"Peak allocated:         {peak_allocated/1e6:.2f} MB\n"
            f"Runtime: {end_time - self.start_time:.2f} seconds"
        )