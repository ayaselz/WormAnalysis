class OpenImageError(Exception):
    def __init__(self, msg: str = "Open Image Error"):
        super.__init__(msg)


class OpenFileError(Exception):
    def __init__(self, msg: str = "Open File Error"):
        super.__init__(msg)


# class PositionFileError(Exception):
#     def __init__(self, msg: str = "Position File Error"):
#         super.__init__(msg)
