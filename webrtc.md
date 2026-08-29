
                    │
                    ▼
              WebRTC APM Thread
                    │
                    ▼
              Processed Queue

from webrtc_audio_processing import AudioProcessing

apm = AudioProcessing(
    enable_ns=True,
    enable_agc=True,
    enable_hpf=True,
    enable_aec=False,
    enable_vad=False,
)

apm.set_stream_format(
    input_sample_rate=16000,
    input_channels=1,
    output_sample_rate=16000,
    output_channels=1,
)

apm.set_ns_level("moderate")


processed_queue = queue.Queue()


def webrtc_processor():

    FRAME_SIZE = 160  # 10 ms @ 16 kHz

    while True:

        with ring_lock:

            if not ring_queue:
                continue

            index, chunk = ring_queue.popleft()

        output = []

        for i in range(0, len(chunk), FRAME_SIZE):

            frame = chunk[i:i + FRAME_SIZE]

            if len(frame) != FRAME_SIZE:
                break

            processed = apm.process_stream(frame)
            output.append(processed)

        processed_chunk = np.concatenate(output)

        processed_queue.put((index, processed_chunk))
