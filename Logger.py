import os
import logging
import inspect

class CustomLogger:
    def __init__(self, log_file_name):
        # Create the 'log' folder if it doesn't exist
        log_folder = 'log'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # Set up the logging configuration
        log_file_path = os.path.join(log_folder, log_file_name)
        logging.basicConfig(filename=log_file_path, level=logging.INFO,
                            format='%(asctime)s ::  %(lineno)d - %(funcName)s ::  %(levelname)s ::  %(message)s')
        self.logger = logging.getLogger(__name__)

    def log_info(self, message, is_duplicate=False, is_constant=False):

        try:
            frame = inspect.stack()[1]
            caller_file = frame[0].f_globals['__file__']
            lineno = frame[0].f_lineno

            if is_duplicate:
                caller_file = 'Auto-ML\data_clean.py'
            elif is_constant:
                caller_file = 'Auto-ML\data_ingestion.py'

            log_message = f"{caller_file} - line {lineno} - INFO - {message}"
            self.logger.info(log_message)
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def log_warning(self, message):
        try:
            frame = inspect.stack()[1]
            caller_file = frame[0].f_globals['__file__']
            lineno = frame[0].f_lineno

            log_message = f"{caller_file} - line {lineno} - WARNING - {message}"
            self.logger.warning(log_message)
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def log_error(self, message):
        try:
            frame = inspect.stack()[1]
            caller_file = frame[0].f_globals['__file__']
            lineno = frame[0].f_lineno

            log_message = f"{caller_file} - line {lineno} - ERROR - {message}"
            self.logger.error(log_message)
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


