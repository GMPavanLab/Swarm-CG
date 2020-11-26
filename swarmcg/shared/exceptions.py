from .styling import sep

header_error = "\n\n-- ! ERROR ! --\n"
header_warning = "\n-- ! WARNING ! --\n"
header_gmx_error = "\n\n  GMX ERROR MSG\n" + sep + "\n\n"

# TODO: disable showing exceptions traceback for distributing to users ? maybe there is a way we can keep it active for our dev purpose though ?

class BaseError(Exception):
    """
    Base exception class.
    """
    def __init__(self, msg):
        self.message = header_error + msg
        super().__init__(self.message)


class ExecError(Exception):
    """
    Base exception class.
    """
    def __init__(self, msg):
        self.message = header_gmx_error + msg
        super().__init__(self.message)


class InvalidArgument(BaseError):

    base_msg = (
        "Cannot interpret argument -{name} as provided: {value}. "
        "{additional_info}"
        "Please check your parameters or look at the help (-h) for an example."
    )

    def __init__(self, argument, value, additional_info=""):
        self.argument = argument
        self.value = value
        self.additional_info = additional_info
        self.message = self.base_msg.format(
            name=argument, value=value, additional_info=additional_info
        )
        super().__init__(self.message)


class IncompleteOptimisationFile(BaseError):
    pass


class OptimisationResultsError(BaseError):
    pass


class MissingCoordinateFile(BaseError):
    pass


class MissingTrajectoryFile(BaseError):
    pass


class MissingItpFile(BaseError):
    pass


class MissingIndexFile(BaseError):
    pass


class InputArgumentError(BaseError):
    pass


class ExecutableNotFound(BaseError):
    pass


class AvoidOverwritingFolder(BaseError):
    pass


class MDSimulationInputError(BaseError):
    pass


class MissformattedFile(BaseError):
    pass


class ComputationError(ExecError):
    pass