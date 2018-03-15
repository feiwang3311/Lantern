Meow Log Tool
=============================

A tool to help you create logger which print to file, console or use `Loggly <https://www.loggly.com/>`_

.. code-block:: python

    import sys
    from meowlogtool import log_util
    loggly_api = "YOUR-API-OF-LOGGLY.COM"

    if __name__ == "__main__":

        # log to console and file
        logger1 = log_util.create_logger("temp_file", print_console=True)
        logger1.info("LOG_FILE") # log using logger1

        # log to console, file and loggly.com
        logger2 = log_util.create_logger("loggly", print_console=True, use_loggly=True, loggly_api_key=loggly_api)
        logger2.info("Log from python")

        # attach log to stdout (print function)
        s1 = log_util.StreamToLogger(logger1)
        sys.stdout = s1

        # anything print to console will be log
        print ('I am Pusheen the cat')
        a = 1234
        print ('I eat 3 shortcakes already. It is too short')
        print ('cost = ', a)
        html_log = log_util.up_gist('doggy.log', 'test_doggy', 'test_doggy')
        print('link on gist %s'%(html_log))



