import time

def measure_execution_time(function_to_measure, *args) -> float:
    start_time = time.time()
    function_to_measure(*args)
    end_time = time.time()
    return end_time-start_time