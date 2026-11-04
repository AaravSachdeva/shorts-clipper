# Changes in this fork

Fork of [Anil-matcha/AI-Youtube-Shorts-Generator](https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator) (MIT). These fixes came out of testing `--mode local` on an hour-long video podcast.

## Fixes

### 1. Long videos crashed during highlight selection (`shorts_generator/highlights.py`)
Videos longer than 30 minutes are split into 20-minute chunks. The prompt showed the LLM **absolute** timestamps, but the code then added the chunk offset again and capped all times at the chunk length. So a model that returned the timestamps it was shown had every highlight in chunks 2+ rejected, and after 3 retries the run died with `RuntimeError: Highlight generator produced invalid output after 3 attempts`.

- Chunk prompts now show chunk-relative timestamps, which matches the offset that gets added back.
- A chunk's allowed duration now includes its 60-second overlap tail, so highlights there aren't cut off.

### 2. Jittery vertical crop (`shorts_generator/local/clipper.py`)
The old cropper ran a Haar face detector on every frame and re-centred on the largest box. False positives on background objects were common, so the crop moved on about 60–70% of frames, by about 13–19 px on average and up to 169 px.

New approach, aimed at static podcast/interview cameras:
1. Detect camera cuts (HSV histogram correlation) and sample faces every 3rd frame on a 640-px-wide downscale.
2. For each shot, choose **one** crop centre: the median of the dominant face cluster, weighted by face area. False positives get outvoted.
3. Hold the crop still for the whole shot and snap to the new position on a cut.

Measured on three clips: 0% of frames shift by more than 1.5 px, versus 59–70% before.

### 3. Speed and output format (`shorts_generator/local/clipper.py`)
- `_cut_subclip` now puts `-ss` before `-i` (input seeking) instead of decoding the source from 0. Rendering 3 clips from a 59-minute episode went from 35 min to 5 min.
- Output is now H.264/yuv420p with `+faststart`, piped straight into ffmpeg. The old OpenCV `mp4v` output (MPEG-4 Part 2) often won't play on phones or in browsers.

### 4. Dependency pins (`requirements-local.txt`)
- `av<17`: PyAV 17+ breaks faster-whisper 1.2.x (`open() got an unexpected keyword argument 'metadata_errors'`).
- `opencv-python<5`: OpenCV 5 removed `CascadeClassifier`.

## `harness/` — test harness
`harness/run.py` runs the real pipeline (`get_highlights()` and the local clipper) with a stand-in LLM. Each prompt the repo builds is saved to `harness/prompts/<hash>.txt`, and the response is read from `harness/answers/<hash>.txt`. This lets you test the pipeline without an API key, and replay a run deterministically.

The included prompts and answers are from a test run on **Cordkillers Ep 377** ("Chekov's White Noise", 2021) by Tom Merritt, Brian Brushwood and guest Iyaz Akhtar, [archived on Internet Archive](https://archive.org/details/cordkillers-ep-377) under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The prompts contain Whisper transcript excerpts of that episode. The highlight picks in `harness/answers/` were written by hand (Claude) following the repo's own virality prompt, not by GPT or Gemini.

To reproduce: put the episode's MP4 at `input/cordkillers377.mp4`, then run:

```bash
python harness/run.py render 3
```
