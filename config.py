# Dependencies
import json


class Config(object):
    config_file = './config.json'

    def __init__(self):
        with open(self.config_file) as f:
            self.settings = json.load(f)


config = Config()

settings = config.settings
