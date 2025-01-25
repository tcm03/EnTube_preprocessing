import inspect

def example_function():
    # Get the current call stack
    stack = inspect.stack()
    print(f'stack: {stack}')
    print(f'type(stack[0]): {type(stack[0])}')
    # The caller is one frame above the current function
    caller_frame = stack[1]
    
    # Extract details about the caller
    caller_filename = caller_frame.filename  # File where the caller resides
    caller_lineno = caller_frame.lineno      # Line number in the caller file
    caller_function = caller_frame.function  # Function name of the caller
    
    # Print the details
    print(f"Caller File: {caller_filename}")
    print(f"Caller Line Number: {caller_lineno}")
    print(f"Caller Function: {caller_function}")

def caller_function():
    # Call the example function
    example_function()

if __name__ == "__main__":
    caller_function()