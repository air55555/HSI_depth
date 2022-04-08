import os
from configparser import ConfigParser

CONFIG_FILE = '{}/config.ini'.format(os.path.dirname(os.path.abspath(__file__)))
config = ConfigParser()
config.read(CONFIG_FILE)
config.set('CONFIG',"start","0")
config.set('CONFIG', "end", "704")
with open(CONFIG_FILE, 'w') as configfile:
    config.write(configfile)

config.read(CONFIG_FILE)
start = int(config['CONFIG']["start"])
end = int(config['CONFIG']["end"])
run_string = config['CONFIG']["run_string"]
mid = int((start+end)/2)


for  i in range(20,300,20):
    os.system(run_string)
    config.set('CONFIG',"start",str(mid-i))
    config.set('CONFIG', "end",str(mid+i))
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)
config.set('CONFIG',"start","0")
config.set('CONFIG', "end", "704")
with open(CONFIG_FILE, 'w') as configfile:
    config.write(configfile)

