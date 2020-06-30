import os
from datetime import datetime


class Printer():
    """
    描述一个将字符串同时输出到终端并保存到文件里的类型
    """
    def __init__(self, file):
        self.file = file
        self.open_or_close = False
        self._check()
        self._open()

    def _check(self):
        """"""
        path, _ = os.path.split(self.file)
        assert os.path.isdir(path)

    def _open(self):
        self.info = open(self.file, 'w')
        self.open_or_close = True

    def _close(self):
        self.info.close()
        self.open_or_close = False

    def pprint(self, text):
        """
        将字符串输出到屏幕或终端，同时将其作为一行写到文件中
        :param text: 将要输出的字符串
        :return:
        """
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        print(time + text)
        self.info.write(time + text + '\n')

def pr():
    """
    there is nothing
    :return:
    """
    pass

if __name__ == '__main__':
    print(pr.__doc__)