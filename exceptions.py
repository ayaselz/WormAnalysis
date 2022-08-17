class OpenImageError(Exception):
    def __init__(self, msg: str = "Open Image Error"):
        super.__init__(msg)
