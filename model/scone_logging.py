""" Contains various functions for logging information."""
def log_metrics(metrics_dict, fileptr=None, experiment=None, index=0):
    """ Logs metrics to a file and a Crayon experiment.

    Inputs:
        metrics_dict (dict str->float): Maps from a string (metric name) to a
            value.
        fileptr (file, optional): Filepointer for the logging file.
        experiment (CrayonExperiment, optional): Experiment to log to Tensorboard.

    Raises:
        ValueError if both fileptr and experiment are None
    """
    if not fileptr and not experiment:
        raise ValueError("Either filename or experiment must not be None for logging.")

    for name, value in metrics_dict.items():
        fileptr.write(name + ":\t" + str(value) + "\n")
        experiment.add_scalar_value(name, value, step=index)
