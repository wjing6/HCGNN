import logging

LOG_FORMAT = "%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s"
logging.basicConfig(format = LOG_FORMAT)
log = logging.getLogger(__name__)