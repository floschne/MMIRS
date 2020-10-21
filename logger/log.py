import logging.config

import os

logging.config.fileConfig(os.path.join(os.getcwd(), 'logger', 'logging.conf'))

# create logger
api_logger = logging.getLogger('API')
backend_logger = logging.getLogger('Backend')
