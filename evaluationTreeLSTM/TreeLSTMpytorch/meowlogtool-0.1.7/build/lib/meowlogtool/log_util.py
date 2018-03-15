import logging
import json
import requests
def up_gist(logger_path, name, description = "meowlogtool", client_id = None, client_secret= None):
    """
    Upload your log to gist
    :param logger_path: path to your text file
    :param name: file name on gist
    :param description: description on gist
    :param client_id: client_id on gist
    :param client_secret: client_secret on gist
    :return:
    """
    """
    {
      "description": "the description for this gist",
      "public": true,
      "files": {
        "file1.txt": {
          "content": "String file contents"
        }
      }
    }
    """
    "?client_id=xxxx&client_secret=yyyy"
    file = open(logger_path, 'r')
    file_content = file.read()
    gist_obj = {}
    gist_obj['description'] = description
    gist_obj['public'] = False
    gist_obj['files'] = {}
    gist_obj['files'][name] = {}
    gist_obj['files'][name]['content'] = file_content
    json_string = json.dumps(gist_obj)
    params = {'client_id': client_id, 'client_secret':client_secret}
    r = requests.post('https://api.github.com/gists', data=json_string, params = params)
    response = json.loads(r.content)
    html_url = response['html_url']
    return html_url


def create_logger(logger_name, print_console = False, use_loggly = False, loggly_api_key = None, loggly_tag = "python"):
    """
    Create a logger write to file logger_name.log
    :param logger_name: name of the file
    :param print_console: (Default False) True = print log on console (also write to file).
    :param use_loggly: (Default False) Set true if you want to use loggly
    :param loggly_api_key (Default None) Put your loggly api key here
    :param loggly_tag: tag for loggly
    :return: logger
    """
    FORMAT = '%(asctime)s : %(levelname)s : %(message)s'
    logFormatter = logging.Formatter(FORMAT)
    logging.basicConfig(filename=logger_name + '.log', level=logging.DEBUG, format=FORMAT)
    logger = logging.getLogger(logger_name)
    if (print_console):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logFormatter)
        logger.addHandler(console_handler)

    if (use_loggly):
        assert loggly_api_key != None
        import loggly.handlers
        lgy = loggly.handlers.HTTPSHandler(
            'https://logs-01.loggly.com/inputs/'+loggly_api_key+'/tag/'+loggly_tag)
        lgy.setFormatter(logFormatter)
        logger.addHandler(lgy)

    return logger


class StreamToLogger(object):
   """
   Source: https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())