import logging

formatter = logging.Formatter("%(name)s:%(levelname)s:%(asctime)s:%(message)s")


def setup_logger(name, level=logging.INFO):
    """To setup as many loggers as you want"""

    # handler = logging.Handler()
    # handler.setFormatter(formatter)

    logging.basicConfig(
        filename="app.log",
        filemode="a",
        level=level,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # logger.addHandler(handler)

    return logger
