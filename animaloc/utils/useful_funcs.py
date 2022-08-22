import os 
import errno
from datetime import date, datetime

def mkdir(path):
    ''' To make a directory from a path '''
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def current_date():
    ''' To get current date in YYYYMMDD format '''
    today = date.today().strftime('%Y%m%d')
    return today

def get_date_time():
    ''' To get current date and time in "d/m/Y H:M:S" format '''
    today = date.today().strftime('%d/%m/%Y')
    now = datetime.now().strftime('%H:%M:%S')
    return today , now

def vdir(obj):
    ''' To filter out 'special methods' of an object '''
    return [m for m in dir(obj) if not m.startswith('__')]