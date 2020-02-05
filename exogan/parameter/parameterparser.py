import configobj

class ParameterParser():

    def __init__(self):
        super().__init__()
        self._read = False
    

    def transform(self,section,key):
        val = section[key]
        newval = val
        if isinstance(val,list):
            try:
                newval = list(map(float,val))

            except:
                pass
        elif isinstance(val, (str)):
            if val.lower() in ['true']:
                newval = True
            elif val.lower() in ['false']:
                newval = False
            else:
                try:
                    newval = float(val)
                except:
                    pass
        section[key]=newval
        return newval


    def read(self,filename):
        import os.path
        if not os.path.isfile(filename):
            raise Exception('Input file {} does not exist'.format(filename))
        self._raw_config = configobj.ConfigObj(filename)
        # print('Raw Config file is {}, filename is {}'.format(self._raw_config,filename))
        self._raw_config.walk(self.transform)
        config = self._raw_config.dict()
        # print('Config file is {}, filename is {}'.format(config,filename))
    
    def generalpars(self):
        config = self._raw_config.dict()
        parameters = config['General']
        return parameters
    
    def trainpars(self):
        config = self._raw_config.dict()
        parameters = config['Training']
        return parameters
    
    def comppars(self):
        config = self._raw_config.dict()
        parameters = config['Completion']
        return parameters