import logging
import colorful

# My logger configuration
class Logger:
    def __init__(self, LEVEL, COLORED=False, TAG_MODULE=None):
        # Logger Configuration
        self.COLORED = COLORED

        if TAG_MODULE is not None:
            self.TAG_MODULE = TAG_MODULE
        else: 
            self.TAG_MODULE = ''

        self.logger = logging.getLogger(__name__)
        self.logger_format = '%(asctime)s  : : %(levelname)s : : %(message)s'
        self.logger_date_format = '[%Y/%m/%d %H:%M:%S %Z]'

        if LEVEL == "DEBUG":
            logging.basicConfig(level=logging.DEBUG, format=self.logger_format, datefmt=self.logger_date_format)
        else:
            logging.basicConfig(level=logging.INFO,  format=self.logger_format, datefmt=self.logger_date_format)
    
    def info(self, message):
        if self.COLORED:
            self.logger.info( colorful.purple(f'{self.TAG_MODULE}: {message}') )
        else:
            self.logger.info(f'{self.TAG_MODULE}: {message}')
    
    def debug(self, message):
        if self.COLORED:
            self.logger.debug( colorful.purple(f'{self.TAG_MODULE}: {message}') )
        else:
            self.logger.debug(f'{self.TAG_MODULE}: {message}')

    def warning(self, message):
        if self.COLORED:
            self.logger.warning( colorful.purple(f'{self.TAG_MODULE}: {message}') )
        else:
            self.logger.warning(f'{self.TAG_MODULE}: {message}')
    
    def error(self, message):
        if self.COLORED:
            self.logger.error( colorful.purple(f'{self.TAG_MODULE}: {message}') )
        else:
            self.logger.error(f'{self.TAG_MODULE}: {message}')
