settings = dict(
    log=dict(author="Boi Mai Quach"),
    bandpass_filter=dict(low_freq=0.1, high_freq=45),
    notch_filter = dict(line_freq=50),
    epochs=dict(start_time=-0.2, end_time=1.000, duration=1),
    ica=dict(ica_n_components = .99, method="picard", decim=None),
    autoreject=dict(threshold=0.25),
)
