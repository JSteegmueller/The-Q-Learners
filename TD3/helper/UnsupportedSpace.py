class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """

    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)
