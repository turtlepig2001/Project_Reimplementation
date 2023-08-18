'''
Date: 2023-08-19 00:43:19
LastEditors: turtlepig
LastEditTime: 2023-08-19 00:43:19
Description:  epochtimer
'''

def epoch_timer(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60)) # 不足一分钟的秒数

    return elapsed_mins, elapsed_secs

