export type SpeechFeatures = {
  energy: number;
  speech_rate: number;
  pitch_variance: number;
};

export function createAudioAnalyzer(stream: MediaStream) {
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  const time = new Uint8Array(analyser.fftSize);
  const freq = new Uint8Array(analyser.frequencyBinCount);
  let crossings = 0;
  let lastSign = 0;
  let lastSampleAt = performance.now();

  function sample(): SpeechFeatures {
    analyser.getByteTimeDomainData(time);
    analyser.getByteFrequencyData(freq);
    let sum = 0;
    let peak = 0;
    for (let i = 0; i < time.length; i++) {
      const centered = (time[i] - 128) / 128;
      sum += centered * centered;
      if (Math.sign(centered) !== lastSign && Math.abs(centered) > 0.02) {
        crossings += 1;
        lastSign = Math.sign(centered);
      }
      peak = Math.max(peak, Math.abs(centered));
    }
    const energy = Math.min(1, Math.sqrt(sum / time.length) * 4);
    const elapsed = Math.max(0.2, (performance.now() - lastSampleAt) / 1000);
    const speechRate = Math.min(1, crossings / elapsed / 12);
    lastSampleAt = performance.now();
    crossings = 0;
    const centroid =
      freq.reduce((acc, value, i) => acc + value * i, 0) /
      Math.max(1, freq.reduce((acc, value) => acc + value, 0));
    const pitchVariance = Math.min(1, Math.abs(centroid - 24) / 80);
    return {
      energy: round(energy),
      speech_rate: round(speechRate * peak),
      pitch_variance: round(pitchVariance),
    };
  }

  function close() {
    void audioContext.close();
  }

  return { sample, close };
}

function round(n: number) {
  return Math.round(n * 100) / 100;
}
