class LoggerNotFoundException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return f"LoggerNotFoundException: {self.message}"

class TrainerFastDevRunException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return f"TrainerFastDevRunException: {self.message}"
