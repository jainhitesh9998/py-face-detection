import inspect
import os
from inspect import getmodule
from pathlib import Path
from shutil import copy2

from py_data import path


class Data:
    def __init__(self):
        self.__home = str(Path.home())

        self.__proj = Path.cwd().parent
        while os.path.isfile(str(self.__proj)+'/__init__.py'):
            self.__proj = self.__proj.parent
        self.__proj = str(self.__proj)

        self.__data_proj = Path(self.__proj + '/proj_data')
        self.__data_home = Path(self.__home + '/.proj_data')

        if not Path.is_dir(self.__data_home):
            Path.mkdir(self.__data_home)
            copy2(path.get('path.py'), self.__data_home)
            copy2(path.get('__init__.py'), self.__data_home)

        if not Path.is_dir(self.__data_proj):
            p = Path(self.__data_proj)
            p.symlink_to(self.__data_home)
            p.resolve()

        print('symlink ' + str(self.__data_proj) + ' set to py-data directory at ' + str(self.__data_home))

    def create_dir(self, dir_name, contents=None):
        module_name = (getmodule(inspect.currentframe().f_back).__name__ + '.').split('.')[0]
        dirs = dir_name.split('/')
        dirs.insert(0, module_name)
        dir = Path(str(self.__data_home))

        for d in dirs:
            d = d.strip()
            if len(d) > 0:
                dir = Path(str(dir) + '/' + d)
                if not Path.is_dir(dir):
                    Path.mkdir(dir, parents=True)
                    copy2(path.get('__init__.py'), str(dir))
                    copy2(path.get('path.py'), str(dir))

        if contents:
            for content in contents:
                copy2(content, str(dir))

        return str(dir)
