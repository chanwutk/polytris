import subprocess


class GRun:
    def __init__(self, instance_name: str):
        self.instance_name = instance_name

    def run(self, command: str):
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)