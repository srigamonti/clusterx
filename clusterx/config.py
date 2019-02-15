# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import os
import json
import numpy as np

class CellConfig():

    def __init__(self,filename="cellconfig.json"):
        cwd = os.getcwd()
        self.configpath = os.path.join(cwd,"cellconfig.json")
        self.config = {}
        self.filename = filename
        self.set_defaults()

    def set_defaults(self):    
        self.config['GENERAL'] = {
            'PRECISION': '8',
            'PBC': {"a1":  True, "a2": True, "a3": True}
        }

    def read(self):    
        try:
            configstr = open(self.configpath,'r').read()
            self.config = json.loads(configstr)[0]
            print(self.config)
        except:
            #print("No %s file present\n"%(self.filename))
            pass

    def write(self):    
        with open(self.configpath, 'w+') as configfile:
            json.dump(self.config,configfile,indent=4)

    def get_pbc(self):
        pbc = self.config['GENERAL']['PBC']
        
        return np.array([pbc['a1'],pbc['a2'],pbc['a3']])

    def is_2D(self):
        pbc = self.get_pbc()
        if (pbc == [True,True,False]).all():
            return True
        else:
            return False
