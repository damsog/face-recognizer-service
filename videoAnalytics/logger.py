import logging

# My logger configuration
class Logger:
    def __init__(self, LEVEL):
        # Logger Configuration
        self.logger = logging.getLogger(__name__)
        self.logger_format = '%(asctime)s  : : %(levelname)s : : %(name)s : : %(message)s'
        self.logger_date_format = '[%Y/%m/%d %H:%M:%S %Z]'

        if LEVEL == "verbose":
            logging.basicConfig(level=logging.DEBUG, format=self.logger_format, datefmt=self.logger_date_format)
        else:
            logging.basicConfig(level=logging.INFO,  format=self.logger_format, datefmt=self.logger_date_format)
    
    def info(self, message):
        self.logger.info(message)
    
    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
