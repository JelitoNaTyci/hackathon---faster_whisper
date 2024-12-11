import pyaudio
from faster_whisper import WhisperModel
import threading
import wave
import time

CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
frames = []
frame = []
threads = []
final_result = []

ARRAY_SEQ = 0
END_POINT = 63 # (RATE * (length of files in sec)) / CHUNK_SIZE = ~4s

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel("tiny", device="cpu", compute_type="int8")

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

print("--- START REC ---")
start_time = time.time()

def read_file(filename, index, final_result):
    segments, info = model.transcribe(filename, beam_size=5)
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        # final_result.insert(index, segment.text) # Index for sorting if not in sequence? 
        final_result.append(segment.text)

def start_threading(filename, index, final_result):
    thread = threading.Thread(target=read_file, args=(filename, index, final_result))
    threads.append(thread)
    thread.start()

def save_into_file(name, array_seq):
    wf = wave.open(name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames[array_seq]))
    wf.close

while True:
    try:
        frame = []
        name = "./output/voice" + str(ARRAY_SEQ) + ".wav"
        for i in range(0, END_POINT):
            data = stream.read(CHUNK_SIZE)
            frame.append(data)

        frames.append(frame.copy())
        save_into_file(name, ARRAY_SEQ)
        start_threading(name, ARRAY_SEQ, final_result)

        ARRAY_SEQ += 1
    except KeyboardInterrupt:
        #Translate rest of cutted stream while interuppted
        frames.append(frame.copy())
        save_into_file(name, ARRAY_SEQ)
        start_threading(name, ARRAY_SEQ, final_result)

        # Wait for all threads are done with transcribing
        for t in threads:
            t.join()

        print("--- REC STOPPED %s seconds ---" % (time.time() - start_time))
        break

stream.stop_stream()
stream.close()
p.terminate()

print(final_result)