import os
import shutil


def GetFilesFromDir(DirPath, filetpye):
    Files = []
    for r, d, f in os.walk(DirPath):
        for file in f:
            if filetpye in file:
                Files.append(os.path.join(r, file))
    return Files


def remove(path):
    if os.path.isdir(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
        else:
            raise ValueError("file {} is not a file or dir.".format(path))
