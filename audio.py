import numpy as np
import pyaudio
from threading import Thread


class Listener:
    def __init__(self):
        """
        controls audio listening by opening up a stream in Pyaudio.
        """
        print("starting listener")

        self.running = True
        self.connected = False
        self.logging = False
        self.mic_in = 0

        # set up mic listening func
        self.CHUNK = 2 ** 11
        self.RATE = 44100
        self.p = pyaudio.PyAudio()

        # get USB mic as source
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=numdevices - 1
        )

    def mainloop(self):
        loop = Thread(target=self.snd_listen)
        loop.start()

    def snd_listen(self):
        """
        Listens to the microphone/ audio input. Logs the intensity/ amplitude
        to hivemind.
        A secondary function it analyses the input sound for a fundamental freq.
        This is currently a redundant function.
        """
        # logging.info("mic listener: started!")
        while self.running:
            data = np.frombuffer(
                self.stream.read(self.CHUNK, exception_on_overflow=False),
                dtype=np.int16,
            )
            peak = np.average(np.abs(data)) * 2
            if peak > 1000:
                bars = "#" * int(50 * peak / 2 ** 16)
                # print("MIC LISTENER: %05d %s" % (peak, bars))

                self.mic_in = peak  # / 30000

                # normalise it for range 0.0 - 1.0
                normalised_peak = ((peak - 0) / (20000 - 0)) * (1 - 0) + 0
                if normalised_peak > 1.0:
                    normalised_peak = 1.0

                # put normalised amplitude into Nebula's dictionary for use
                self.mic_in = normalised_peak
            else:
                self.mic_in = 0

    def terminate(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()