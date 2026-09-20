from audiomentations import (
    AddColorNoise,
    AddGaussianNoise,
    BandStopFilter,
    ClippingDistortion,
    Gain,
    Mp3Compression,
    Normalize,
    PitchShift,
    SevenBandParametricEQ,
    Shift,
    TimeStretch,
)

# power decay (dB/octave) has to be hard coded here bc of how PSD falls off w frequency
_F_DECAY = {"pink_noise": -3.01, "brown_noise": -6.02}


def build_aug(name, cfg):
    if name == "gain_shift":
        limit = cfg["limit"]
        return Gain(min_gain_db=-limit, max_gain_db=limit, p=1.0)

    if name == "normalization":
        return Normalize(p=1.0)

    if name == "clipping":
        return ClippingDistortion(
            min_percentile_threshold=cfg["min_percentile"],
            max_percentile_threshold=cfg["max_percentile"],
            p=1.0,
        )

    if name == "white_noise":
        return AddGaussianNoise(
            min_amplitude=cfg["min_amplitude"],
            max_amplitude=cfg["max_amplitude"],
            p=1.0,
        )

    if name in _F_DECAY:
        decay = _F_DECAY[name]
        return AddColorNoise(
            min_snr_db=cfg["min_snr_db"],
            max_snr_db=cfg["max_snr_db"],
            min_f_decay=decay,
            max_f_decay=decay,
            p=1.0,
        )

    if name == "time_stretch":
        return TimeStretch(
            min_rate=cfg["min_rate"],
            max_rate=cfg["max_rate"],
            leave_length_unchanged=True,
            p=1.0,
        )

    if name == "time_shift":
        frac = cfg["max_fraction"]
        return Shift(
            min_shift=-frac,
            max_shift=frac,
            shift_unit="fraction",
            rollover=True,
            p=1.0,
        )

    if name == "pitch_shift":
        semitones = cfg["semitones"]
        return PitchShift(min_semitones=-semitones, max_semitones=semitones, p=1.0)

    if name == "eq_filtering":
        gain = cfg["gain_db"]
        return SevenBandParametricEQ(min_gain_db=-gain, max_gain_db=gain, p=1.0)

    if name == "bandstop_filter":
        return BandStopFilter(
            min_center_freq=cfg["min_center_hz"],
            max_center_freq=cfg["max_center_hz"],
            p=1.0,
        )

    if name == "mp3_compress":
        return Mp3Compression(
            min_bitrate=cfg["min_bitrate"],
            max_bitrate=cfg["max_bitrate"],
            p=1.0,
        )

    return None
