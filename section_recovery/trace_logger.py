import logging

TRACE_LEVEL_NUM = 5


class TraceLogger(logging.getLoggerClass()):
    def __init__(self, name, level=logging.NOTSET):
        super(logging.getLoggerClass(), self).__init__(name, level)

        logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

    def trace(self, message, *args, **kws):
        # Yes, logger takes its '*args' as 'args'.
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kws)
