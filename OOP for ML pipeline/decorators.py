import time

def timing_log(func):
    """
    A decorator that logs the time a function takes to execute.
    
    Parameters:
    - func (callable): The function to be decorated.
    
    Returns:
    - callable: The wrapped function with timing logging.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        runtime = end_time - start_time  # Calculate the runtime
        print(f"Function '{func.__name__}' took {runtime:.4f} seconds to run")
        return result  # Return the function's result
    return wrapper