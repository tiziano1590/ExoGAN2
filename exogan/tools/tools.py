import numpy as np
import gzip
import pickle
import h5py
import os

class ReportInterface(object):

    # ...more details about this class...

    @classmethod
    def __save_dict_to_hdf5__(cls, dic, filename):
        """..."""
        if os.path.exists(filename):
            raise ValueError('File %s exists, will not overwrite.' % filename)
        with h5py.File(filename, 'w') as h5file:
            cls.__recursively_save_dict_contents_to_group__(h5file, '/', dic)

    @classmethod
    def __recursively_save_dict_contents_to_group__(cls, h5file, path, dic):
        """..."""
        # argument type checking
        if not isinstance(dic, dict):
            raise ValueError("must provide a dictionary")
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(h5file, h5py._hl.files.File):
            raise ValueError("must be an open h5py file")
        # save items to the hdf5 file
        for key, item in dic.items():
            try:
                key = "%d" % int(key)
            except:
                key = str(key)
            if not isinstance(key, str):
                raise ValueError("dict keys must be strings to save to hdf5")
            # save strings, numpy.int64, and numpy.float64 types
            if isinstance(item, (np.int64, np.float64, str)):
                h5file[path + key] = item
                if not h5file[path + key][()] == item:
                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')
            # save numpy arrays
            elif isinstance(item, np.ndarray):
                h5file[path + key] = item
                if not np.array_equal(h5file[path + key][()], item):
                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')
            # save dictionaries
            elif isinstance(item, dict):
                cls.__recursively_save_dict_contents_to_group__(h5file, path + key + '/', item)
            # other types cannot be saved and will result in an error
            else:
                raise ValueError('Cannot save %s type.' % type(item))

    @classmethod
    def __load_dict_from_hdf5__(cls, filename):
        """..."""
        with h5py.File(filename, 'r') as h5file:
            return cls.__recursively_load_dict_contents_from_group__(h5file, '/')

    @classmethod
    def __recursively_load_dict_contents_from_group__(cls, h5file, path):
        """..."""
        ans = {}
        for key, item in h5file[path].items():
            try:
                key = "%d" % int(key)
            except:
                key = str(key)

            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = cls.__recursively_load_dict_contents_from_group__(h5file, path + key + '/')
        return ans