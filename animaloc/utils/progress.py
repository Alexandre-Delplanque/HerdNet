from typing import Iterable

def progressbar(
    iterable: Iterable, 
    prefix: str = '', 
    suffix: str = '', 
    decimals: int = 1, 
    length: int = 100, 
    fill: str = '█', 
    printEnd: str = "\r"
    ) -> None:
    '''

    Call in a loop to create terminal progress bar

    Args::
        iterable (iterable): iterable to loop through
        prefix (str, optional): prefix string
        suffix (str, optional): suffix string
        decimals (int, optional): positive number of decimals in percent complete
        length (int, optional): character length of bar
        fill (str, optional): bar fill character
        printEnd (str, optional): end character (e.g. "\r", "\r\n")
    '''

    total = len(iterable)

    def print_progress(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)

    print_progress(0)

    for i, item in enumerate(iterable):
        yield item
        print_progress(i + 1)

    print()