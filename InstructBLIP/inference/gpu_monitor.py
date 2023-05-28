DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

import subprocess
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]


import time
import sys
from loguru import logger
logger.add(sys.stderr, filter="my_module")

def gpu_monitor(
    gpu_idxs: list,
    threshold: int=0,
    metric: str="utilization.gpu",
    max_idle_sec: int=60,
    monitor_iterval: int=3,
    info_iterval: int=60
    ):
    """Moniting GPU Utils.
    Args:
    - gpu_idxs: list, index for gpu, start from 0.
    - threshold: int, threshold for gpu utils.
    - metric: str, metric for gpu utils, ref to `DEFAULT_ATTRIBUTES`.
    - max_idle_sec: int, maximum idle time in seconds.
    - monitor_iterval: int, iterval in seconds for refreshing the utils info.
    - info_iterval: int, iterval in seconds for infoing not die.
    Return:
    - True, when all gpu utils <= `threshold` for `max_idle_sec`.
    Example:
    ```
        IDLE = gpu_monitor(
            gpu_idxs=[0,1], threshold=10, metric="utilization.gpu",
            max_idle_sec=60, monitor_iterval=3, info_iterval=60
        )
    ```
    """
    assert metric in DEFAULT_ATTRIBUTES, f"{metric} not in {DEFAULT_ATTRIBUTES}"
    idle_sec, info_sec=0, 0
    logger.warning(f"Monitor started. [GPUs: {gpu_idxs}] [Threshold: {threshold}] "\
        f"[Max Idle Sec: {max_idle_sec}]")

    while True:
        gpu_info = get_gpu_info()
        gpu_utils = [int(gpu_info[i][metric]) for i in gpu_idxs]
        if any([utils<=threshold for utils in gpu_utils]):
            idle_sec+=monitor_iterval
            logger.info(f"Idled for {idle_sec} seconds. "\
                f"[GPU_utils: {dict(zip(gpu_idxs, gpu_utils))}] [Threshold: {threshold}]")
        elif idle_sec!=0:
            idle_sec=0
            logger.warning(f"\n\nIdled time reset. "\
                f"[GPU_utils: {dict(zip(gpu_idxs, gpu_utils))}] [Threshold: {threshold}]\n")

        if idle_sec>=max_idle_sec:
            logger.warning(f"Idled for {idle_sec} seconds.")
            return True

        info_sec += monitor_iterval
        if info_sec >= info_iterval:
            info_sec = 0
            logger.warning("Moniting...")
        time.sleep(monitor_iterval)


from slack import WebClient
class SlackMessenger:
    def __init__(self, slack_token, channel_id):
        self.client = WebClient(token=slack_token)
        self.channel_id = channel_id

    def send_message(self, message_text):
        response = self.client.chat_postMessage(
            channel=self.channel_id, text=message_text)

        if response["ok"]:
            print("Message sent successfully!")
        else:
            print("Failed to send message. Error:", response["error"])


if __name__ == "__main__":

    gpu_idxs=[0,1,2,3]
    slack_token = "xo"+"xb-4592272045554-5329966721573-lm2jRDs6hspW9rRoLe3OrQO8" # 'Notify Bot' token
    channel_id = "notification"
    message_text = "GPU available."

    if gpu_monitor(
        gpu_idxs, threshold=10, metric="utilization.gpu",
        max_idle_sec=60, monitor_iterval=3, info_iterval=60
        ):
        messenger = SlackMessenger(slack_token, channel_id)
        messenger.send_message(message_text)

    # nohup /root/anaconda3/envs/PlayGround/bin/python gpu_monitor.py > /root/Documents/DEMOS/InstructBLIP/inference/nohup.log& 2>&1