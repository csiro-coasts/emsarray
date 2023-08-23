class CommandException(RuntimeError):
    """Raise this to exit a command line script gracefully.

    See Also
    --------
    :func:`.utils.console_entrypoint`
    """
    def __init__(self, message: str, code: int = 1):
        self.message = message
        self.code = code
