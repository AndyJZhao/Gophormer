import os
import sys

sys.path.append((os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))
from utils.exp.summarizer import summarize_exp
from utils.functions import get_cur_time

import time
from timeloop import Timeloop
from datetime import timedelta
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--minutes', type=int, default=10, help='')
    args = parser.parse_args()

    summarize_exp()
    tl = Timeloop()

    @tl.job(interval=timedelta(minutes=args.minutes))
    def regular_commit():
        summarize_exp()
        print('-' * 30 + f'\n\nSummarized at {get_cur_time()}')

    tl.start()
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            tl.stop()
            break