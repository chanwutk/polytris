import os
from multiprocessing import Process


DIR = './pipeline-stages/video-masked'


def process(file: str):
    filename = os.path.join(DIR, file)
    output_filename = f"{filename[:-len('.mp4')]}.x264.mp4"

    if os.path.exists(output_filename):
        os.system(output_filename)

    command = (
        "docker run --rm -v $(pwd):/config linuxserver/ffmpeg " +
        "-i {input_file} ".format(input_file=os.path.join('/config', filename)) +
        "-vcodec libx264 " +
        "{output_file}".format(output_file=os.path.join('/config', output_filename))
    )
    print(command)
    os.system(command)


def main():
    ps: list[Process] = []
    for file in os.listdir(DIR):
        p = Process(target=process, args=(file,))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
        p.terminate()


if __name__ == '__main__':
    main()