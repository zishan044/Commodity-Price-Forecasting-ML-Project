import sys

class CustomException(Exception):
    def __init__(self, error: Exception, sys_module: sys):
        _, _, exc_tb = sys_module.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        error_message = f"Error occurred in file: {file_name}, line: {line_number}, message: {str(error)}"

        super().__init__(error_message)
        self.error_message = error_message

    def __str__(self):
        return self.error_message
