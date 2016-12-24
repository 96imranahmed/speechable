import pyaudio
import sys
import threading
import wave
import os
import time

#pyaudio Constants
CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

#Speech buffer Constants
BUFFER = 40

#Pause Constants
cur_pause = False
do_end = False

#Other Constants
output_dir = os.getcwd()+"/output/"

def chk_pause():
    global cur_pause, do_end, output_dir
    while do_end == False:
        input_var = input()
        if input_var == "":
            cur_pause = True
        elif input_var == "clean":
            [os.remove(output_dir + f) for f in os.listdir("./output") if f.endswith(".wav") ]
            print("Audio files cleaned!")
    sys.exit(0)
    
def run_record():
    global cur_pause
    frames = []
    while True:
        if (len(frames) < int(RATE/ CHUNK* BUFFER)):
            frames.append(stream.read(CHUNK))
            if len(frames) == (int(RATE/ CHUNK* BUFFER) - 1): #Buffer now full!
                frames.append(stream.read(CHUNK))
            if cur_pause:
                save(frames)
                cur_pause = False
        else:
            del frames[0]
            frames.append(stream.read(CHUNK))
            if cur_pause:
                save(frames)
                cur_pause = False

def save(frames_in):
    print("SAVED")
    file_pth = output_dir + str(int(time.time()*1000))+'.wav'
    wf = wave.open(file_pth, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames_in))
    wf.close()
    pass


def main(argv):
    global stream, p, do_end
    t = None
    try:
        t = threading.Thread(target=chk_pause)
        t.start()
        run_record()
    except KeyboardInterrupt:
        print("Press enter to kill...")
        do_end = True
        stream.stop_stream()
        stream.close()
        p.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main(sys.argv)
