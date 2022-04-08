import os
from configparser import ConfigParser

CONFIG_FILE = '{}\config.ini'.format(os.path.dirname(os.path.abspath(__file__)))
config = ConfigParser()
config.read(CONFIG_FILE)
config.set('CONFIG',"start","0")
config.set('CONFIG', "end", "704")
config.set('CONFIG',"mid","300")
with open(CONFIG_FILE, 'w') as configfile:
    config.write(configfile)

config.read(CONFIG_FILE)
start = int(config['CONFIG']["start"])
end = int(config['CONFIG']["end"])
mid = int(config.get('CONFIG',"mid"))
run_string = config['CONFIG']["run_string"]
mid_calculated = int((start+end)/2)


for j in range(mid-abs(mid-mid_calculated),mid+abs(mid-mid_calculated),10):
    mid_temp=j
    for  i in range(20,300,20):

        config.set('CONFIG',"start",str(mid_temp-i))
        config.set('CONFIG', "end",str(mid_temp+i))
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        print(run_string)
        os.system(run_string)
config.set('CONFIG',"mid","300")
config.set('CONFIG',"start","0")
config.set('CONFIG', "end", "704")
with open(CONFIG_FILE, 'w') as configfile:
    config.write(configfile)

